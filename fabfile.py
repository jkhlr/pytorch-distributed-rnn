import json
import time
from pathlib import Path

from fabric import task
from patchwork.transfers import rsync

MASTER = '192.168.2.15'
SLAVES = [
    '192.168.2.2',
    '192.168.2.3',
    '192.168.2.4',
    '192.168.2.5',
    '192.168.2.6',
    '192.168.2.7',
    '192.168.2.8',
    '192.168.2.9',
    '192.168.2.10',
    '192.168.2.11',
    '192.168.2.12'
]

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
            '--stacked-layer': 1,
            '--hidden-units': 32,
            '--dropout': 0,
            '--batch-size': batch_size,
            '--epochs': 6
        }
    }
    for batch_size in [32, 64, 128, 256]
    for num_slots in [1, 2, 3, 4]
    for num_hosts in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    for trainer in ['local', 'distributed', 'horovod']
    if trainer != 'local' or num_hosts == num_slots == 1
]

DEBUG_RUN = {
    'trainer': 'distributed',
    'hosts': 2,
    'slots': 1,
    'parameters': {
        '--stacked-layer': 1,
        '--hidden-units': 32,
        '--dropout': 0,
        '--batch-size': 32,
        '--epochs': 6
    }
}


@task
def prepare_connections(c):
    keyscan_command = f'ssh-keyscan {" ".join(SLAVES)}'
    known_hosts_file = '~/.ssh/known_hosts'

    new_entries = c.run(keyscan_command).stdout
    existing_entries = c.run(f'cat {known_hosts_file}').stdout
    if new_entries not in existing_entries:
        c.run(f'{keyscan_command} >> {known_hosts_file}')

    hostnames = c.run(f'mpirun --host {",".join(SLAVES)} hostname').stdout
    if not all(hostname == result for hostname, result in
               zip(SLAVES, hostnames.splitlines())):
        raise ValueError("Can't connect to slaves")


@task
def copy_src(c):
    source_path = f'./src/{DATASET}'
    dest_path = str(SRC_DIR)

    c.run(f'mkdir -p {dest_path}')
    rsync(c, source_path, dest_path, delete=True)


@task
def install_wheels(c):
    # keep trailing slash to copy contents of source_path into dest_path
    source_path = '/path/to/wheels/'
    dest_path = '/tmp/wheels'

    c.run(f'mkdir -p {dest_path}')
    rsync(c, source_path, dest_path, delete=True)

    pip_bin = BIN_DIR / 'pip'
    c.run(f'{pip_bin} install {dest_path}/*.whl')


def run_training_configuration(
        connection,
        trainer,
        parameters,
        num_hosts,
        slots_per_host
):
    host_string = ','.join(
        f'{address}:{slots_per_host}'
        for address in ([MASTER] + SLAVES)[:num_hosts]
    )
    parameter_string = ' '.join(
        f'{name} {value}'
        for name, value
        in parameters.items()
    )
    python_bin = BIN_DIR / 'python'

    if trainer == 'local':
        assert (num_hosts == slots_per_host == 1)
        command = f'{venv_python} {TRAIN_SCRIPT} {parameter_string} local'
    elif trainer == 'distributed':
        command = (
            f'mpirun --bind-to none --map-by slot '
            f'-np {num_hosts * slots_per_host} '
            f'--host {host_string} '
            f'{venv_python} {TRAIN_SCRIPT} {parameter_string} distributed'
        )
    elif trainer == 'horovod':
        horovodrun_bin = BIN_DIR / 'horovodrun'
        command = (
            f'{horovodrun_bin} '
            f'-np {num_hosts * slots_per_host} '
            f'--hosts {host_string} '
            f'{venv_python} {TRAIN_SCRIPT} {parameter_string} horovod'
        )
    else:
        raise ValueError(f'Invalid trainer: {trainer}')

    stdout, stderr, seconds = measure_time(connection, command)
    return command, stdout, stderr, seconds


def measure_time(connection, command):
    start_time = time.time()
    result = connection.run(command)
    end_time = time.time()
    execution_seconds = end_time - start_time
    return result.stdout, result.stderr, execution_seconds


def run_training(connection, configurations=None, result_filename=None):
    result_filename = result_filename or RESULT_FILE
    configurations = configurations or TRAIN_RUNS
    results = []
    for run in configurations:
        command, stdout, stderr, seconds = run_training_configuration(
            connection=connection,
            trainer=run['trainer'],
            parameters=run['parameters'],
            num_hosts=run['hosts'],
            slots_per_host=run['slots']
        )
        results.append({
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            'seconds': seconds,
            'config': run
        })
        with open(result_filename, 'w') as f:
            json.dump({'results': results}, f)


@task
def run_debug(c):
    run_training(
        connection=c,
        configurations=[DEBUG_RUN],
        result_filename='results_debug.json'
    )


@task
def run_all(c):
    run_training(connection=c)
