import pytest
import argparse
import copy

from nvmitten.configurator.fields import *


def test_field_parse():
    f1 = Field("f1")
    args = parse_fields([f1], [])
    assert "f1" not in args
    assert len(args) == 0

    args = parse_fields([f1], ["--f1", "hello world"])
    assert "f1" in args
    assert args["f1"] == "hello world"
    assert len(args) == 1


def test_field_from_string():
    f1 = Field("f1", from_string=bool)
    f2 = Field("f2", from_string=int)
    f3 = Field("f3", from_string=lambda s: len(s))

    args = parse_fields([f1, f2, f3], ["--f2", "1337", "--f3", "123456"])
    assert "f1" not in args
    assert args["f2"] == 1337
    assert args["f3"] == 6

    args = parse_fields([f1, f2, f3], ["--f1"])
    assert len(args) == 1
    assert args["f1"] == True


def test_field_copy():
    f1 = Field("f1", from_string=bool)
    cp = copy.copy(f1)
    assert cp is f1


def test_field_deepcopy():
    f1 = Field("f1", from_string=bool)
    cp = copy.deepcopy(f1)
    assert cp is f1
