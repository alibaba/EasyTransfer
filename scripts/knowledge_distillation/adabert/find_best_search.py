import os
import re
import sys
import shutil


if __name__ == "__main__":
    task = sys.argv[1]
    exp_id = sys.argv[2]

    with open("./logs/{}_{}.log".format(task.lower(), exp_id)) as f:
        s = f.read()
    best_step = 0
    best_acc = 0
    for step, acc in re.findall(r"global step (\d+): Acc = (0.\d+)", s):
        step = int(step)
        acc = float(acc)
        if acc > best_acc:
            best_step = step
            best_acc = acc
    print(best_step, best_acc)
    model_dir = "models/{}/adabert_{}/search/".format(task.upper(), exp_id)
    best_dir = os.path.join(model_dir, "best")
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    shutil.copy(os.path.join(model_dir, "model.ckpt-%d.index" % best_step),
                os.path.join(best_dir, "model.ckpt-%d.index" % best_step))
    shutil.copy(os.path.join(model_dir, "model.ckpt-%d.meta" % best_step),
                os.path.join(best_dir, "model.ckpt-%d.meta" % best_step))

    shutil.copy(os.path.join(model_dir, "model.ckpt-%d.data-00000-of-00002" % best_step),
                os.path.join(best_dir, "model.ckpt-%d.data-00001-of-00002" % best_step))
    shutil.copy(os.path.join(model_dir, "model.ckpt-%d.data-00000-of-00002" % best_step),
                os.path.join(best_dir, "model.ckpt-%d.data-00000-of-00002" % best_step))

    shutil.copy(os.path.join(model_dir, "wemb_%d.npy" % best_step),
                os.path.join(best_dir, "wemb.npy"))
    shutil.copy(os.path.join(model_dir, "pemb_%d.npy" % best_step),
                os.path.join(best_dir, "pemb.npy"))
    shutil.copy(os.path.join(model_dir, "arch_%d.json" % best_step),
                os.path.join(best_dir, "arch.json"))

    with open(os.path.join(best_dir, "checkpoint"), "w") as f:
        f.write('model_checkpoint_path: "model.ckpt-%d"' % best_step)

