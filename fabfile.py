import json
import time

from fabric import task

RESULT_FILE = 'results.json'
SLAVES = ['slave']
MASTER = 'master'
TRAIN_SCRIPT = '~/src/corona/main.py'

DEBUG_RUN = {
    'hosts': 2,
    'slots': 1,
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
        'hosts': 1,
        'slots': 1,
        'parameters': {
            '--stacked-layer': 1,
            '--hidden-units': 32,
            '--dropout': 0,
            '--batch-size': 128,
            '--epochs': 1
        }
    },
    {
        'hosts': 1,
        'slots': 2,
        'parameters': {
            '--stacked-layer': 1,
            '--hidden-units': 32,
            '--dropout': 0,
            '--batch-size': 128,
            '--epochs': 1
        }
    },
    {
        'hosts': 2,
        'slots': 1,
        'parameters': {
            '--stacked-layer': 1,
            '--hidden-units': 32,
            '--dropout': 0,
            '--batch-size': 128,
            '--epochs': 1
        }
    },
    {
        'hosts': 2,
        'slots': 2,
        'parameters': {
            '--stacked-layer': 1,
            '--hidden-units': 32,
            '--dropout': 0,
            '--batch-size': 128,
            '--epochs': 1
        }
    }
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


def run_training_configuration(connection, parameters, num_hosts, slots_per_host):
    host_string = ','.join(([MASTER] + SLAVES)[:num_hosts] * slots_per_host)
    parameter_string = ' '.join(
        f'{name} {value}'
        for name, value
        in parameters.items()
    )
    command = f'mpirun --host {host_string} python {TRAIN_SCRIPT} {parameter_string}'
    stdout, stderr, seconds = measure_time(connection, command)
    return command, stdout, stderr, seconds


def measure_time(connection, command):
    start_time = time.time()
    result = connection.run(command, hide='both')
    end_time = time.time()
    execution_seconds = end_time - start_time
    return result.stdout, result.stderr, execution_seconds


@task
def run_training(c):
    results = []
    for run in TRAIN_RUNS:
        command, stdout, stderr, seconds = run_training_configuration(
            connection=c,
            parameters=run['parameters'],
            num_hosts=run['hosts'],
            slots_per_host=run['slots']
        )
        results.append({
            'command': command,
            'stdout': stdout,
            'stderr': stderr,
            'seconds': seconds
        })
    with open(RESULT_FILE, 'w') as f:
        json.dump({'results': results}, f)

@task
def run_debug_training(c):
    command, stdout, stderr, seconds = run_training_configuration(
        connection=c,
        parameters=DEBUG_RUN['parameters'],
        num_hosts=DEBUG_RUN['hosts'],
        slots_per_host=DEBUG_RUN['slots']
    )
    print('COMMAND:')
    print(command)
    print('STDOUT:')
    print(stdout.strip())
    print('STDERR:')
    print(stderr.strip())
    print('SECONDS:')
    print(seconds)