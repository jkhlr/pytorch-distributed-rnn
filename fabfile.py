import json
from pathlib import Path

from fabric import task
from patchwork.transfers import rsync

HOSTS = [
    '10.42.0.50',
    '10.42.0.29',
    '10.42.0.105',
    '10.42.0.56',
    '10.42.0.180',
    '10.42.0.235',
    '10.42.0.244',
    '10.42.0.239',
    '10.42.0.191',
    '10.42.0.41',
    '10.42.0.190',
    '10.42.0.69'
]

DEBUG_RUN = {
    'trainer': 'distributed',
    'hosts': 12,
    'slots': 1,
    'parameters': {
        '--batch-size': 1440,
        '--epochs': 1,
        '--stacked-layer': 2,
        '--hidden-units': 32,
        '--dropout': 0.3,
        '--seed': 123456789
    }
}

WORK_DIR = Path('~/susml/jakob_torben')
BIN_DIR = WORK_DIR / 'bin'
SRC_DIR = WORK_DIR / 'src'
DATASET = 'motion'
TRAIN_SCRIPT = SRC_DIR / DATASET / 'main.py'
RESULT_FILE = 'results.json'

TRAIN_RUNS = [
    {
        'trainer': trainer,
        'hosts': num_hosts,
        'slots': num_slots,
        'parameters': {
            # Should be a multiple of 96, to make training on
            # 1, 2, 4, 8, and 12 nodes with 1, 2 and 4 slots reproducible
            '--batch-size': 1440,
            '--epochs': 10,
            '--stacked-layer': 2,
            '--hidden-units': 32,
            '--dropout': 0.3,
            '--seed': 123456789
        }
    }
    for num_slots in [1, 2, 4]
    for num_hosts in [1, 2, 4, 8, 12]
    for trainer in ['local', 'distributed', 'horovod']
    if trainer != 'local' or num_hosts == num_slots == 1
]


@task
def prepare_connections(c):
    keyscan_command = f'ssh-keyscan {" ".join(HOSTS)}'
    known_hosts_file = '~/.ssh/known_hosts'

    new_entries = c.run(keyscan_command).stdout
    existing_entries = c.run(f'cat {known_hosts_file}').stdout
    if new_entries not in existing_entries:
        c.run(f'{keyscan_command} >> {known_hosts_file}')

    hostnames = c.run(f'mpirun --host {",".join(HOSTS)} hostname').stdout
    if not all(hostname == result for hostname, result in
               zip(HOSTS, hostnames.splitlines())):
        raise ValueError("Can't connect to slaves")


@task
def copy_src(c):
    source_path = f'./src/{DATASET}'
    dest_path = str(SRC_DIR)

    c.run(f'mkdir -p {dest_path}')
    rsync(c, source_path, dest_path, delete=True,
          ssh_opts="-i ~/.ssh/pi_cluster -o StrictHostKeyChecking=no")


@task
def install_wheels(c):
    # keep trailing slash to copy contents of source_path into dest_path
    source_path = './wheels/'
    dest_path = '/tmp/wheels'

    c.run(f'mkdir -p {dest_path}')
    rsync(c, source_path, dest_path, delete=True,
          ssh_opts="-i ~/.ssh/pi_cluster -o StrictHostKeyChecking=no")

    pip_bin = BIN_DIR / 'pip'
    c.run(f'{pip_bin} install {dest_path}/*.whl')


@task
def install_profiler(c):
    pip_bin = BIN_DIR / 'pip'
    c.run(f'{pip_bin} install memory_profiler')


def run_training_configuration(
        connection,
        trainer,
        parameters,
        num_hosts,
        slots_per_host,
        hosts,
        bin_dir,
        train_script
):
    host_string = ','.join(
        f'{address}:{slots_per_host}'
        for address in hosts[:num_hosts]
    )
    parameter_string = ' '.join(
        f'{name} {value}'
        for name, value
        in parameters.items()
    )
    python_bin = Path(bin_dir) / 'python'

    if trainer == 'local':
        assert (num_hosts == slots_per_host == 1)
        command = f'{python_bin} {train_script} {parameter_string} local'
    elif trainer == 'distributed':
        command = (
            'mpirun --bind-to none --map-by slot '
            f'-np {num_hosts * slots_per_host} '
            f'--host {host_string} '
            f'{python_bin} {train_script} {parameter_string} distributed'
        )
    elif trainer == 'horovod':
        # command = (
        #     'mpirun --bind-to none --map-by slot '
        #     '-x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH '
        #     '-mca pml ob1 -mca btl ^openib '
        #     f'-np {num_hosts * slots_per_host} '
        #     f'--host {host_string} '
        #     f'{python_bin} {train_script} {parameter_string} horovod'
        # )
        command = (
            'horovodrun '
            f'-np {num_hosts * slots_per_host} '
            f'--hosts {host_string} '
            f'{python_bin} {train_script} {parameter_string} horovod'
        )
    else:
        raise ValueError(f'Invalid trainer: {trainer}')

    print(command)
    result = connection.run(command)
    return command, result.stdout, result.stderr


def run_training(
        connection,
        hosts=None,
        bin_dir=None,
        train_script=None,
        configurations=None,
        result_filename=None
):
    hosts = hosts or HOSTS
    bin_dir = bin_dir or BIN_DIR
    train_script = train_script or TRAIN_SCRIPT
    configurations = configurations or TRAIN_RUNS
    result_filename = result_filename or RESULT_FILE
    results = []
    for run in configurations:
        command, stdout, stderr = run_training_configuration(
            connection=connection,
            trainer=run['trainer'],
            parameters=run['parameters'],
            num_hosts=run['hosts'],
            slots_per_host=run['slots'],
            hosts=hosts,
            bin_dir=bin_dir,
            train_script=train_script
        )
        results.append({
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            'config': run
        })
        with open(result_filename, 'w') as f:
            json.dump({'results': results}, f)


@task
def run_all(c):
    run_training(connection=c, hosts=HOSTS)


@task
def run_debug(c):
    run_training(
        connection=c,
        hosts=HOSTS,
        configurations=[DEBUG_RUN],
        result_filename='results_debug.json'
    )


@task
def run_debug_docker(c):
    docker_debug_run = {
        'trainer': 'distributed',
        'hosts': 2,
        'slots': 1,
        'parameters': {
            '--batch-size': 1440,
            '--epochs': 1,
            '--stacked-layer': 2,
            '--hidden-units': 32,
            '--dropout': 0.3,
            '--seed': 123456789
        }
    }
    hosts = ['master', 'slave']
    bin_dir = '/usr/local/bin'
    train_script = f'~/src/{DATASET}/main.py'
    result_filename = 'results_debug.json'

    run_training(
        connection=c,
        hosts=hosts,
        bin_dir=bin_dir,
        train_script=train_script,
        configurations=[docker_debug_run],
        result_filename=result_filename
    )
