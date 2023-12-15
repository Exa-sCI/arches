from functools import update_wrapper
from inspect import signature
from time import perf_counter

import numpy as np


class ScopedTimer:
    def __init__(self, log_message="", timer_on=True, **kwargs):
        self.log_message = log_message
        self.print = timer_on

    def __enter__(self):
        self.t0 = perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.t1 = perf_counter()
        if self.print:
            print(self.log_message + f" | Execution took {self.t1-self.t0:0.3e} seconds")


class LogTimer:
    def __init__(self, func, config_map, report_on):
        update_wrapper(self, func)
        self.func = func
        self.func_sig = signature(func)
        self.config_func = config_map[0]
        self.config_msg = config_map[1]

        self.log = {}
        ## can be ('call'), ('exit'), ('call', 'exit'), would be nice to register scoped callbacks or event triggers
        ## if empty tuple, log will be stored but only printed if asked for
        self.report_on = report_on

    def __call__(self, *args, **kwargs):
        ## determine call configuration
        bound_args = self.func_sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        config = self.config_func(**bound_args.arguments)

        ## run and time
        t0 = perf_counter()
        self.func(*args, **kwargs)
        t1 = perf_counter()

        if "call" in self.report_on:
            print(
                self.__name__
                + f" ran with config {self.config_msg(*config)} and executed in {t1-t0:0.3e} seconds"
            )

        ## store in log
        if config in self.log.keys():
            self.log[config].append(t1 - t0)
        else:
            self.log[config] = [t1 - t0]

    def __del__(self):
        if "exit" in self.report_on:
            self.print_log()

    def print_log(self):
        for k, v in self.log.items():
            N_runs = len(v)
            avg_time = np.mean(v)
            std_time = np.std(v)
            print(
                self.__name__
                + f" ran with config {self.config_msg(*k)} {N_runs} times,"
                + f" executing in an average of {avg_time:0.2e} ± {std_time:0.2e} s"
            )

    def clear_log(self):
        self.log = {}

    @property
    def report_on(self):
        return self._report_on

    @report_on.setter
    def report_on(self, val):
        if val in ((None,), ("call",), ("exit",), ("call", "exit"), ("exit", "call")):
            self._report_on = val
        else:
            raise ValueError(
                "Value must be one of ('call'), ('exit'), ('call', 'exit'), or (None,)"
            )


def logtimer(config_map, report_on=("exit",)):
    def _logtimer(func):
        return LogTimer(func, config_map, report_on)

    return _logtimer


if __name__ == "__main__":

    def example_1():
        config_map = (lambda a, b, **kwargs: (a, b), lambda a, b: f"(A: {a}, B: {b})")

        @logtimer(config_map)
        def f(a, b, c):
            return a + b + c

        f.report_on = ("call", "exit")
        f(1, 2, 3)
        f(1, 2, 3)
        f(1, 2, 3)
        f(1, 2, 3)
        f(1, 2, 3)
        f(1, 2, 3)
        f(1, 2, 3)

        f(2, 3, 4)
        f(2, 3, 4)
        f(2, 3, 4)
        f(2, 3, 4)

        f(5, 6, 2)
        f(5, 6, 2)

        f(11, 12, 23)

    def example_2():
        config_map = (lambda a, b, **kwargs: (a, b), lambda a, b: f"(A: {a}, B: {b})")

        @logtimer(config_map)
        def f(a, b, c):
            return a + b + c

        f.report_on = (None,)
        f(1, 2, 3)
        f(1, 2, 3)

        f(2, 3, 4)
        f(2, 3, 4)
        f.print_log()
        f.clear_log()

        print("-----")
        f(5, 6, 2)
        f(5, 6, 2)

        f(11, 12, 23)
        f.print_log()

    example_1()
    print("\n ######### \n")
    example_2()
