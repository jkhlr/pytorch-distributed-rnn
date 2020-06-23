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

WORKDIR = Path('~/susml/jakob_torben')
SRC_DIR = WORKDIR / 'src/'
DATASET = 'motion'
TRAIN_SCRIPT = SRC_DIR / DATASET / 'main.py'
RESULT_FILE = 'results.json'

DEBUG_RUN = {
    'hosts': 2,
    'slots': 2,
    'threads': 2,
    'parameters': {
        '--stacked-layer': 1,
        '--hidden-units': 32,
        '--dropout': 0,
        '--batch-size': 128,
        '--epochs': 1
    }
}

TRAIN_RUNS = [
    {
        'hosts': num_hosts,
        'slots': num_slots,
        'threads': num_threads,
        'parameters': {
            '--stacked-layer': 1,
            '--hidden-units': 32,
            '--dropout': 0,
            '--batch-size': batch_size,
            '--epochs': 6
        }
    }
    for batch_size in [32, 64, 128, 256]
    for num_threads in [1, 2, 3, 4]
    for num_slots in [1, 2, 3, 4]
    for num_hosts in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    if num_slots * num_threads <= 4
]


@task
def prepare_connections(c):
    keyscan_command = f'ssh-keyscan {" ".join(SLAVES)}'
    known_hosts_file = '~/.ssh/known_hosts'

    new_entries = c.run(keyscan_command).stdout
    existing_entries = c.run(f'cat {known_hosts_file}').stdout
    if new_entries not in existing_entries:
        c.run(f'{keyscan_command} >> {known_hosts_file}')

    hostnames = c.run(f'mpirun --host {",".join(SLAVES)} hostname').stdout
    if not all(hostname == result for hostname, result in zip(SLAVES, hostnames.splitlines())):
        raise ValueError("Can't connect to slaves")


@task
def copy_files(c):
    c.run(f'mkdir -p {WORKDIR}')

    source_path = f'./src/{DATASET}'
    rsync(c, source_path, WORKDIR, delete=True)


def run_training_configuration(connection, parameters, num_hosts, slots_per_host, threads_per_slot):
    host_string = ','.join(([MASTER] + SLAVES)[:num_hosts] * slots_per_host)
    parameter_string = ' '.join(
        f'{name} {value}'
        for name, value
        in parameters.items()
    )
    venv_python = WORKDIR / 'bin/python'
    command = (
        f'mpirun --host {host_string} --map-by socket:pe={threads_per_slot} '
        f"{venv_python} {TRAIN_SCRIPT} {parameter_string}'"
    )

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
            parameters=run['parameters'],
            num_hosts=run['hosts'],
            slots_per_host=run['slots'],
            threads_per_slot=run['threads']
        )
        results.append({
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            'seconds': seconds,
            'batch_size': run['parameters']['--batch-size'],
            'nodes': run['hosts'],
            'processes_per_node': run['slots'],
            'threads_per_process': run['threads']
        })
        with open(result_filename, 'w') as f:
            json.dump({'results': results}, f)


@task
def run_debug(c):
    run_training(c, configurations=[DEBUG_RUN], result_filename='results_debug.json')


@task
def run_all(c):
    run_training(c)
