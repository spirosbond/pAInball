# # from __future__ import print_function # Only Python 2.x
from subprocess import Popen, PIPE
from multiprocessing import Process, Array
import time
import json
from enum import Enum
from neural_ntw import NeuralNetwork
from nn_genetics import NNGenetics
from numpy import array, random
import os
import psutil
import random
import gc
from datetime import datetime
import sys
from nn_visualization import DrawNN
import re


class Action(Enum):
    # Helper Class for all AI Actions (outputs)
    nothing = 0
    # LF_Ext = 1
    # LF_Ret = 2
    # RF_Ext = 3
    # RF_Ret = 4
    # Plunger_Press = 5
    # Plunger_Release = 6
    LF_Hit = 1
    RF_Hit = 2

    @classmethod
    def to_string(self, v):
        return str(v) + "\r\n"


class Fbk(Enum):
    # Helper Class for all game feedback
    Mode = 0
    Speed = 1
    Xpos = 2
    Ypos = 3
    Xdir = 4
    Ydir = 5
    Score = 6
    BallCollisions_L = 7
    BallCollisions_R = 8
    BallDrainFlag = 9


def window_id(proc_id):
    # Helper function to get window id
    proc = Popen(
        ["wmctrl", "-lp"], env=os.environ, stdout=PIPE, universal_newlines=True
    )
    out = proc.communicate()[0]
    for l in out.split("\n"):
        s = l.strip().split()
        if len(s) > 1 and int(s[2]) == proc_id:
            return s[0]


def wait_for_window(proc_id, timeout=3):
    # Helper function to open window
    wid = None
    poll_interval = 0.1
    attempts = max(1, timeout // poll_interval)
    while wid is None and attempts > 0:
        attempts -= 1
        wid = window_id(proc_id)
        if wid is None:
            proc = psutil.Process(proc_id)
            for child_proc in proc.children(recursive=True):
                wid = window_id(child_proc.pid)
                if wid is not None:
                    break
        if wid is None:
            time.sleep(poll_interval)
    return wid


def resize_window(pid, geometry):
    # Helper function to resize window
    wid = wait_for_window(pid)
    if wid is None:
        print(f"could not get window for process ID: {pid}")
    else:
        cmd = ["wmctrl", "-i", "-r", wid, "-e", geometry]
        Popen(cmd, env=os.environ)


def get_feedback(p, i, fbk, scores, ball_collisions_l, ball_collisions_r, balls_used):
    """Process that receives the feedback of the game via stdout
    i: Unique identifier of the process
    fbk: Shared array that has the live feedback of the game
    scores: Shared array to capture all scores
    ball_collisions: Shared array to capture all ball collisions
    balls_used: Shared array to capture all lost balls
    """
    ball_collisions_l[i] = 0
    ball_collisions_r[i] = 0
    scores[i] = 0
    balls_used[i] = 0
    new_ball_flag = 0
    for line in iter(p.stdout.readline, ""):
        try:
            # print(line)
            for idx, v in enumerate(line.split(",")):
                fbk[idx] = float(v)
            # print(fbk[0:])
            # print(fbk[Fbk.Score.value])
            if fbk[Fbk.Score.value] >= 0:
                scores[i] = fbk[Fbk.Score.value]
            if new_ball_flag == 0 and fbk[Fbk.BallDrainFlag.value] == 1:
                # print(fbk[Fbk.BallDrainFlag.value])
                balls_used[i] += 1
                new_ball_flag = 1
            elif fbk[Fbk.BallDrainFlag.value] == 0:
                new_ball_flag = 0
            # scores[i] = 0
            # ball_collisions[i] = 0
            ball_collisions_l[i] = fbk[Fbk.BallCollisions_L.value]
            ball_collisions_r[i] = fbk[Fbk.BallCollisions_R.value]
            # print(f"{scores[i]},{ball_collisions[i]},{balls_used[i]}")
        except:
            # p.stdout.flush()
            continue
        # p.stdout.flush()
        # gc.collect()


def press_keys(p, i, fbk, nn, timeout):
    """Process that sends the commands to the game via stdin
    p: Game Process
    i: Unique identifier of the process
    fbk: Shared array that has the live feedback of the game (from get_feedback)
    nn: The neural network to control this game
    timeout: Game length duration
    """
    while fbk[Fbk.Mode.value] != 1:
        # print(f"sleeping")
        time.sleep(0.1)
    start = time.time()
    while (fbk[Fbk.Mode.value] == 1) and (time.time() - start) < timeout:
        outputs = nn.think(fbk[1:6])
        # print(
        #     f"NN_{i} input: {fbk[1:6]} output: {outputs[-1]} action: {Action(array(outputs[-1]).argmax())}"
        # )
        p.stdin.write(Action.to_string(array(outputs[-1]).argmax()))
        p.stdin.flush()
        # gc.collect()
        time.sleep(0.15)
        # p.stdin.write(Action.LF_Ret.value)
        # p.stdin.flush()
        # time.sleep(1)
        # print("sending key")
    # print(f"press_keys {i} finished")


def start_game(
    i,
    nn,
    scores,
    ball_collisions_l,
    ball_collisions_r,
    game_durations,
    balls_used,
    window_coords,
    timeout,
):
    """Process that launches multiple instances of the game
    i: Unique identifier of the process
    nn: The neural network to control this game
    scores: Shared array to capture all scores
    ball_collisions: Shared array to capture all ball collisions
    game_durations: Shared array to capture the duration of the game
    balls_used: Shared array to capture all lost balls
    window_coords: The wondow coordinations for this process
    timeout: Game length duration
    """
    feedback = Array("f", len(Fbk))
    with Popen(
        ["bin/SpaceCadetPinball"],
        stdin=PIPE,
        stdout=PIPE,
        stderr=PIPE,
        text=True,
        universal_newlines=True,
    ) as p:
        p1 = Process(
            target=get_feedback,
            args=(
                p,
                i,
                feedback,
                scores,
                ball_collisions_l,
                ball_collisions_r,
                balls_used,
            ),
        )
        p1.start()

        p2 = Process(target=press_keys, args=(p, i, feedback, nn, timeout))
        start = time.time()
        p2.start()

        # print(f"start_game {i} started with id {p.pid}")
        time.sleep(2)
        resize_window(p.pid, window_coords)
        # p1.join()
        p2.join()
        game_durations[i] = time.time() - start
        # print(f"Game duration: {game_durations[i]}")
        p1.terminate()
        # print(f"start_game {i} finished")


def create_next_generation(parents, n_of_ch, n_of_rand_ch, mut_prob, genetics):
    """Function that creates the children for the next generation
    parents: List of NNs to be used as parents
    n_of_ch: Number of children to generate
    n_of_rand_ch: Number of random children to generate
    mut_prob: Probability of a mutation occuring
    genetics: The genetics class to use to create the next generation
    """
    child_seed = genetics.crossover_average(parents)
    # child_seed = genetics.crossover_shuffle(parents)
    children = parents
    children.append(child_seed)

    for i in range(n_of_ch - len(children) - n_of_rand_ch):
        children.append(genetics.mutate(child_seed, mut_prob))
    for i in range(n_of_rand_ch):
        children.append(
            NeuralNetwork(
                parents[0].num_of_inputs,
                parents[0].num_of_outputs,
                parents[0].hidden_layers,
            )
        )

    return child_seed, children


if __name__ == "__main__":
    random.seed()
    num_of_nns = 43  # Set the number of Neural Network Children per epoch
    num_of_rand_children = 3  # Number for random Neural Networks per epoch

    num_of_inputs = len(Fbk) - 5
    num_of_outputs = len(Action)
    hidden_layers = [7, 7]  # Number of hiden layers and their nodes
    start_timeout = 30
    n_of_parents = 2
    mut_prob = 0.09
    # Starting game duration per epoch. It is increasing by 10 sec every 10 epoch

    nn_genetics = NNGenetics()

    children = []
    if len(sys.argv) > 1:
        parents = []
        for arg_id in range(1, len(sys.argv)):
            print(f"Loading parent to start from: {sys.argv[arg_id]}")
            parent = NeuralNetwork(num_of_inputs, num_of_outputs, hidden_layers)
            parent.load_weights(sys.argv[arg_id])
            parents.append(parent)

        path = sys.argv[1].split("/")
        directory = f"{path[0]}/{path[1]}/{path[2]}"
        match = re.search(r"ep(\d+)_", path[3])
        start_epoch = int(match.group(1)) + 1
        child_seed, children = create_next_generation(
            parents, num_of_nns, num_of_rand_children, mut_prob, nn_genetics
        )
        # for i in range(num_of_nns-len(children)-num_of_rand_children):
        # children.append(nn_genetics.mutate(child, 0.01))
        # for i in range(num_of_rand_children):
        # children.append(NeuralNetwork(num_of_inputs, num_of_outputs, hidden_layers))
    else:
        now = datetime.now()
        now_str = now.strftime("%Y-%m-%d %H_%M_%S")

        directory = (
            f"AIPlayer/runs/nn{num_of_nns}_p{n_of_parents}_pr{mut_prob}_{now_str}"
        )
        os.makedirs(directory)
        start_epoch = 0
        for i in range(num_of_nns):
            children.append(NeuralNetwork(num_of_inputs, num_of_outputs, hidden_layers))

    print(directory)
    draw_nn = DrawNN(directory)

    scores = Array("f", num_of_nns)
    ball_collisions_l = Array("f", num_of_nns)
    ball_collisions_r = Array("f", num_of_nns)
    game_durations = Array("f", num_of_nns)
    balls_used = Array("f", num_of_nns)

    print(f"starting from", end=" ")
    for epoch in range(start_epoch, 100000):
        print(f"epoch: {epoch} ")
        timeout = start_timeout + (start_timeout / 3) * (epoch // 10)
        nns = []
        processes = []

        for i, child in enumerate(children):
            # neural_network = NeuralNetwork(num_of_inputs, num_of_outputs, hidden_layers)
            # child.print_weights()
            nns.append(child)

            # outputs = neural_network1.think(array([1,1,1]))
            # neural_network1.print_outputs(outputs)
            sizeX = 300
            sizeY = 220
            windowsPerRow = (
                2400 // sizeX
            )  # 2400px is based on the width of my screen 1920x1080. Adjust per your preference
            winLocationX = (i * sizeX) % 2400 + 100
            winLocationY = (i // windowsPerRow) * (sizeY + 40) + 25
            g = Process(
                target=start_game,
                args=(
                    i,
                    child,
                    scores,
                    ball_collisions_l,
                    ball_collisions_r,
                    game_durations,
                    balls_used,
                    f"0,{winLocationX},{winLocationY},{sizeX},{sizeY}",
                    timeout,
                ),
            )
            g.start()
            processes.append(g)
            time.sleep(0.2)

        print(f"Waiting for games to end...")
        for i in range(num_of_nns):
            processes[i].join()

        # print(f"Best nn: {array(scores).argmax()}")
        # print(f"Best nn: {scores[array(scores).argmax()]}")

        print(f"Calculating fitness...")
        parent_ids, fitness = nn_genetics.fitness(
            array(scores),
            array(ball_collisions_l),
            array(ball_collisions_r),
            array(game_durations),
            array(balls_used),
            n_of_parents=n_of_parents,
        )
        # fitness = [round(f / timeout, 2) for f in fitness]
        nn_genetics.save_fitness(epoch, parent_ids, fitness, f"{directory}/fitness.csv")
        parent_nns = []
        for i, idx in enumerate(parent_ids):
            print(f"Parent {i}: {idx} with fitness: {fitness[i]}")
            parent_nns.append(nns[idx])
            nns[idx].save_weights(
                f"{directory}/ep{epoch}_parent_{i}_id{idx}_{fitness[i]}.csv"
            )
            draw_nn.configure(
                [num_of_inputs, 7, 7, num_of_outputs], nns[idx].get_weights()
            )
            draw_nn.save_draw(epoch, i, idx, fitness[i])

        for ch in children:
            del ch

        print(f"Creating next generation...")
        child_seed, children = create_next_generation(
            parent_nns, num_of_nns, num_of_rand_children, mut_prob, nn_genetics
        )

        for nn in nns:
            del nn
        # draw_nn.configure(
        #     [num_of_inputs, 7, 7, num_of_outputs], child_seed.get_weights()
        # )
        # draw_nn.save_draw(epoch, str(parent_ids), str(parent_ids), "n/a")

        # child = nn_genetics.crossover(parent_nns)

        # for i in range(num_of_nns-len(children)-num_of_rand_children):
        #     children.append(nn_genetics.mutate(child, 0.01))
        # for i in range(num_of_rand_children):
        #     children.append(NeuralNetwork(num_of_inputs, num_of_outputs, hidden_layers))
        # for child in children:
        #     child.print_weights()
        gc.collect()
