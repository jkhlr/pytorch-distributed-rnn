import json
import os
from pathlib import Path
from random import shuffle

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

DOCKER_HOSTS = [
    'master',
    'slave'
]

DEBUG_RUN = {
    'trainer': 'local',
    'hosts': 1,
    'slots': 1,
    'parameters': {
        '--epochs': 1,
        '--seed': 123456789,
        '--no-validation': ''
    }
}

WORK_DIR = Path('~/susml/jakob_torben')
BIN_DIR = WORK_DIR / 'bin'
SRC_DIR = WORK_DIR / 'src'
DATASET = 'motion'
TRAIN_SCRIPT = SRC_DIR / DATASET / 'main.py'
RESULT_FILE = 'results.json'
RESULT_FILE_NETWORK = 'results_network.json'

TRAIN_RUNS = [
    {
        'trainer': trainer,
        'hosts': num_hosts,
        'slots': 1,
        'parameters': {
            # Batch size should be a multiple of 96, to make training on
            # 1, 2, 4, 8, and 12 nodes with 1, 2 and 4 slots reproducible
            '--batch-size': batch_size,
            '--epochs': 1,
            '--seed': 123456789,
            '--no-validation': ''
        }
    }
    for batch_size in [480, 960, 1440]
    for num_hosts in [1, 2, 4, 8, 12]
    for trainer in ['local', 'distributed', 'horovod']
    if trainer != 'local' or num_hosts == 1
]


def test_mpi_connection(c, hosts):
    keyscan_command = f'ssh-keyscan {" ".join(hosts)}'
    known_hosts_file = '~/.ssh/known_hosts'

    new_entries = c.run(keyscan_command).stdout
    existing_entries = c.run(f'cat {known_hosts_file}', warn=True).stdout
    if new_entries not in existing_entries:
        c.run(f'{keyscan_command} >> {known_hosts_file}')
    c.run(f'mpirun --host {",".join(hosts)} hostname')


@task
def prepare_connections(c):
    test_mpi_connection(c, HOSTS)


@task
def prepare_connections_docker(c):
    test_mpi_connection(c, DOCKER_HOSTS)


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
    rsync(
        c,
        source_path,
        dest_path,
        delete=True,
        ssh_opts="-i ~/.ssh/pi_cluster -o StrictHostKeyChecking=no"
    )

    pip_bin = BIN_DIR / 'pip'
    c.run(f'{pip_bin} install {dest_path}/*.whl')


@task
def install_profiler(c):
    pip_bin = BIN_DIR / 'pip'
    c.run(f'{pip_bin} install memory_profiler')


@task
def run_network_test(c):
    run_with_tc(c)


def run_with_tc(c, hosts=None, result_file=None):
    hosts = hosts or HOSTS
    results_file = result_file or RESULT_FILE_NETWORK
    args = dict(
        num_hosts=12,
        slots_per_host=1,
        parameters={
            '--batch-size': 1440,
            '--epochs': 1,
            '--seed': 123456789,
            '--no-validation': ''
        },
        bin_dir=BIN_DIR,
        train_script=TRAIN_SCRIPT,
        hosts=hosts or HOSTS
    )
    # rules_delay = [
    #     ('delay', time, 'ms') for time in [0, 1, 2, 5, 10, 25, 50, 100, 200, 300, 400]
    # ]
    rules_loss = [
        ('loss', percent, '%') for percent in [15]
    ]
    rules = rules_loss  # + rules_loss

    for i, (rule_type, rule_value, rule_unit) in enumerate(rules):
        rule = f'{rule_type} {rule_value}{rule_unit}'
        # if i == 0:
        #     set_rule(c, rule, hosts, delete=True)
        print(f'Setting rule {rule}')
        set_rule(c, rule, hosts)
        print(f'Running with rule {i}/{len(rules)}')
        print(rule)
        for trainer in ['distributed', 'horovod']:
            command = get_command(trainer=trainer, **args)
            print(command)
            stdout, stderr = run_command(c, command, warn=True)
            result = {
                'trainer': trainer,
                'rule_type': rule_type,
                'rule_value': rule_value,
                'rule': rule,
                'stdout': stdout,
                'stderr': stderr,
                'command': command
            }
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)['results']
            else:
                results = []
            with open(results_file, 'w') as f:
                results.append(result)
                json.dump({'results': results}, f)
        set_rule(c, rule, hosts, delete=True)


def set_rule(c, rule, hosts, delete=False):
    action = 'delete' if delete else 'add'
    for host in hosts:
        command = f'ssh pi@{host} "sudo tc qdisc {action} dev eth0 root netem {rule}"'
        print(command)
        c.run(command)


def get_command(
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
        horovod_bin = Path(bin_dir) / 'horovodrun'
        command = (
            f'{horovod_bin} --start-timeout 300 '
            f'-np {num_hosts * slots_per_host} '
            f'--hosts {host_string} '
            f'{python_bin} {train_script} {parameter_string} horovod'
        )
    else:
        raise ValueError(f'Invalid trainer: {trainer}')

    return command


def run_command(connection, command, warn=False):
    print(command)
    result = connection.run(command, warn=warn)
    return result.stdout, result.stderr


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
    result_filename = result_filename or RESULT_FILE

    configurations = list(configurations or TRAIN_RUNS)
    shuffle(configurations)

    for i, run in enumerate(configurations):
        command = get_command(
            trainer=run['trainer'],
            parameters=run['parameters'],
            num_hosts=run['hosts'],
            slots_per_host=run['slots'],
            hosts=hosts,
            bin_dir=bin_dir,
            train_script=train_script
        )
        with open(result_filename, 'r') as f:
            results = json.load(f)['results']

        executed_commands = [result['command'] for result in results]
        if command in executed_commands:
            print(f'Skipping configuration {i}/{len(configurations)}')
            continue

        print(f'Running configuration {i}/{len(configurations)}')
        stdout, stderr = run_command(connection, command)

        result = {
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            'config': run
        }

        results.append(result)
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
def run_debug_docker(c, trainer):
    docker_debug_run = {
        'trainer': trainer,
        'hosts': 2,
        'slots': 1,
        'parameters': {
            '--epochs': 1,
            '--seed': 123456789,
            '--no-validation': ''
        }
    }
    hosts = DOCKER_HOSTS
    bin_dir = '/usr/local/bin'
    train_script = f'~/src/{DATASET}/main.py'
    result_filename = 'results_debug_docker.json'

    run_training(
        connection=c,
        hosts=hosts,
        bin_dir=bin_dir,
        train_script=train_script,
        configurations=[docker_debug_run],
        result_filename=result_filename
    )
