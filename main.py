import torch.multiprocessing as mp

import argparse

from actor import actor_process
from learner import learner_process
mp.set_start_method('spawn', force=True)


def run():
    mp.freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_actors', type=int, default=1)
    args = parser.parse_args()
    mp_manager = mp.Manager()
    shared_dict = mp_manager.dict()

    # learner process
    processes = [mp.Process(
        target=learner_process,
        args=(args.n_actors, shared_dict))]

    for actor_id in range(args.n_actors):
        processes.append(mp.Process(
            target=actor_process,
            args=(actor_id, args.n_actors, shared_dict, 'cuda:'+str(actor_id%2+1))))

    for i in range(len(processes)):
        processes[i].start()

    for p in processes:
        p.join()


if __name__ == '__main__':
    run()


