# Copyright (c) Meta Platforms, Inc. and affiliates.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import dataclasses
import json
from argparse import ArgumentParser
from dataclasses import fields, is_dataclass
from typing import Any, Dict, get_origin, List, Optional, Set, Type, TypeVar, Union


T = TypeVar("T")

FALSY_STRINGS = {"off", "false", "0"}
TRUTHY_STRINGS = {"on", "true", "1"}
SEPARATOR = "."
MISSING: str = "???"


class NOTSET:
    pass


class MissingArg(Exception):
    pass


class WrongConfName(Exception):
    pass


class WrongConfType(Exception):
    pass


class WrongArgType(Exception):
    pass


class WrongFieldType(Exception):
    pass


class OptionalDataClass(Exception):
    pass


class DefaultDataClassValue(Exception):
    pass


class ParseJsonAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        try:
            setattr(namespace, self.dest, json.loads(values))
        except json.JSONDecodeError as e:
            parser.error(f"Error decoding JSON: {e}")


def bool_flag(s: str) -> Union[bool, str]:
    """
    Parse boolean arguments from the command line.
    """
    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    elif s == NOTSET:
        return MISSING
    else:
        raise argparse.ArgumentTypeError("Invalid value for a boolean flag!")


def is_optional(some_type: Any) -> bool:
    return some_type == Optional[some_type]


def get_opt_type(some_type: Any) -> Any:
    if is_optional(some_type):
        # allowing Optional dataclasses now: we only recurse in the cli if the name appears in the args
        # if is_dataclass(some_type.__args__[0]):
        #     raise OptionalDataClass("Optional dataclasses not supported")
        return some_type.__args__[0]
    return some_type


def NOCLI(default: Optional[Any] = None) -> Any:
    return dataclasses.field(default=default, metadata={"NOCLI": True})


def is_nocli(some_field: Any) -> bool:
    return (
        isinstance(some_field, dataclasses.Field)
        and some_field.metadata.get("NOCLI", False) is True
    )


def _get_default_value(field_: dataclasses.Field, default_instance: Optional["T"]):
    """We check first for default_instance.
    If not there we check for field.default_factory() then for field.default
    If the default_value is dataclasses.MISSING we normalize it to our MISSING
    """
    default_value = (
        (
            #  dataclasses._DefaultFactory[typing.Any]]` is not a function.
            field_.default_factory()
            if field_.default_factory != dataclasses.MISSING
            else field_.default
        )
        if default_instance is None
        else getattr(default_instance, field_.name)
    )
    if default_value == dataclasses.MISSING:
        # we normalize to MISSING
        return MISSING
    return default_value


def from_cli(
    cls: Type[T],
    param_dict: Dict[str, Any],
    prefix: str = "",
    default_instance: Optional["T"] = None,
    allow_incomplete: Optional[bool] = False,
) -> T:
    """
    Converts a flat dot separated dictionary into a class object.
    The dictionary must come from a CLI (`parse_args`) output directly.
    """
    assert is_dataclass(cls), f"cls must be a dataclass type, got {type(cls).__name__}"
    assert default_instance is None or isinstance(default_instance, cls)
    kwargs = {}

    # to crash/or do some warning if some unused args
    used_args: Set[str] = {"cfg"}
    for field in fields(cls):
        fullname = f"{prefix}{field.name}"
        field_type = get_opt_type(field.type)
        assert allow_incomplete or (fullname not in param_dict) == is_nocli(field)
        cli_value = NOTSET
        if fullname in param_dict:
            cli_value = param_dict[fullname]
            used_args |= {k for k in param_dict if k.startswith(fullname)}

        # default value is field is not set in the CLI
        default_value = _get_default_value(field, default_instance)

        # this field should not be in the CLI, so we use the default value
        if is_nocli(field):
            kwargs[field.name] = default_value

        # this is a dataclass
        elif is_dataclass(field_type):
            # If optional, default value must be None
            # In this case, we only recurse if some params start with our full name
            must_recurse = any(
                [x.startswith(fullname) and y != NOTSET for x, y in param_dict.items()]
            )
            if not must_recurse and is_optional(field.type):
                # Either default_value is None and nothing else was set in the CLI
                # Or it might be set in the default instance which is captured by default_value
                kwargs[field.name] = default_value
            else:
                # dataclass no specified, use a default value
                if cli_value == NOTSET:
                    sub_conf = None if default_value == MISSING else default_value

                # dataclass specified using a named config
                elif isinstance(cli_value, str) and cli_value != MISSING:
                    raise WrongConfName(
                        f"Unknown conf key {cli_value} for field {fullname}"
                    )

                # unexpected type (should not be reachable if `param_dict` is the
                # output of a CLI, as the argument should be of type `str`).
                else:
                    raise WrongArgType(f"Value for {fullname} should be a string!")

                # check that the current config has a correct type
                if sub_conf is not None and not isinstance(sub_conf, field_type):
                    raise WrongConfType(
                        f"Invalid configuration. Provided a configuration of type "
                        f'"{type(sub_conf).__name__}", expected "{field_type.__name__}".'
                    )

                # if we specified a.b = some_name and a.b.c = 5, we want to overwrite some_name.c
                kwargs[field.name] = from_cli(
                    field_type,
                    param_dict=param_dict,
                    prefix=f"{fullname}.",
                    default_instance=sub_conf,
                )

        # this is not a dataclass, with a CLI provided value. try to parse it
        elif cli_value != NOTSET:
            try:
                if get_origin(field_type) is dict:
                    kwargs[field.name] = dict(cli_value) if cli_value else None  # type: ignore
                else:
                    kwargs[field.name] = field_type(cli_value)
            except ValueError as e:
                raise WrongArgType(e)

        # this is not a dataclass, and it does not appear in the CLI
        else:
            kwargs[field.name] = default_value

    # all arguments should be used and not MISSING
    unused_args = {
        k: v
        for k, v in param_dict.items()
        if (k not in used_args) and k.startswith(prefix)
    }
    if unused_args:
        raise RuntimeError(
            f"Some fields in from_cli are unused to instanciate {cls}: {unused_args}"
        )
    for x, y in kwargs.items():
        if y == MISSING:
            raise MissingArg(f"Parameter {prefix}{x} is MISSING")

    return cls(**kwargs)  # type: ignore


def to_cli(
    cls: Type[T], prefix: str = "", parser: Optional[ArgumentParser] = None
) -> ArgumentParser:
    assert is_dataclass(cls), f"cls must be a dataclass type, got {type(cls).__name__}"
    # initialize parser
    if parser is None:
        parser = ArgumentParser(allow_abbrev=False)
        parser.add_argument("--cfg", type=str)

    # for each field in the dataclass
    for field in fields(cls):
        # sanity check / get field name / field type
        if prefix == "":
            assert field.name != "cfg", "'cfg' field is reserved for cli parser"
        fullname = f"{prefix}{field.name}"
        field_type = get_opt_type(field.type)

        # this field is not allowed in the CLI -> nothing to do
        if is_nocli(field):
            pass

        # dataclass -> recursively add arguments to the CLI
        elif is_dataclass(field_type):
            to_cli(field_type, prefix=f"{fullname}.", parser=parser)
            parser.add_argument(f"--{fullname}", type=str, default=NOTSET)

        elif get_origin(field_type) is dict:
            parser.add_argument(f"--{fullname}", action=ParseJsonAction, type=str)

        # standard parameter
        else:
            parser.add_argument(
                f"--{fullname}",
                type=bool_flag if field_type == bool else field_type,
                help="" if field.metadata is None else field.metadata.get("help"),
                default=NOTSET,
            )
    return parser


def parse_args(cls: Type[T], args: Optional[List[str]] = None) -> T:
    return from_cli(cls, vars(to_cli(cls).parse_args(args)))
