import sys

from slugai.callback import CallbackAggregate, Progress


def test_epoch_start_callback(capsys):
    cb = Progress()
    cb.before_epoch_start(1)
    captured = capsys.readouterr()
    assert captured.out == "Epoch 1 -------------------------\n"


def test_callback_aggregate(capsys):
    cb = CallbackAggregate([Progress()])
    cb.epoch_started(epoch=1)
    captured = capsys.readouterr()
    assert captured.out == "Epoch 1 -------------------------\n"


def test_myoutput(capsys):  # or use "capfd" for fd-level
    print("hello")
    sys.stderr.write("world\n")
    captured = capsys.readouterr()
    assert captured.out == "hello\n"
    assert captured.err == "world\n"
    print("next")
    captured = capsys.readouterr()
    assert captured.out == "next\n"
