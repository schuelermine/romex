from __future__ import annotations
from enum import Enum, auto
from typing import Protocol, cast, final
from fractions import Fraction
from collections.abc import Sequence, Mapping, Iterable
from dataclasses import dataclass, field
from itertools import groupby
import re


__all__ = (
    "Options",
    "NESTED_SUBTRACTION_STACK_MODE",
    "SINGLE_VALUE_SUBTRACTION_MODE",
    "I",
    "V",
    "X",
    "L",
    "C",
    "D",
    "M",
    "DOTS1",
    "DOTS2",
    "DOTS3",
    "DOTS4",
    "DOTS5",
    "S",
    "SILIQUA",
    "SCRIPULUM",
    "DIMIDIA_SEXTULA",
    "SEXTULA",
    "SICILICUS",
    "SEMUNCIA",
    "FIVE_HUNDRED",
    "THOUSAND",
    "FIVE_THOUSAND",
    "TEN_THOUSAND",
    "FIFTY_THOUSAND",
    "HUNDRED_THOUSAND",
    "VinculumKind",
    "Vinculum",
    "mk_roman_sequence",
    "get_roman_value",
)


class SequenceEvaluationMode(Enum):
    NESTED_SUBTRACTION_STACK_MODE = auto()
    SINGLE_VALUE_SUBTRACTION_MODE = auto()


NESTED_SUBTRACTION_STACK_MODE, SINGLE_VALUE_SUBTRACTION_MODE = SequenceEvaluationMode


@dataclass(frozen=True)
class Options:
    """
    Represents the mode for evaluating sequences
    """

    mode: SequenceEvaluationMode = field(default=NESTED_SUBTRACTION_STACK_MODE)
    strict: bool = field(kw_only=True, default=False)
    grouped: bool = field(kw_only=True, default=True)


DEFAULT_OPTIONS = Options()


class HasRomanValue(Protocol):
    def roman_value(self, options: Options = ..., /) -> int | Fraction:
        ...


class BasicRomanSigil(Enum):
    I = 1
    V = 5
    X = 10
    L = 50
    C = 100
    D = 500
    M = 1000

    def roman_value(self, _options: Options = DEFAULT_OPTIONS, /) -> int:
        """
        Gets the fractional or integer value this object represents
        """
        return self.value

    def __str__(self) -> str:
        return f"{self.name}"


class FractionalRomanSigil(Enum):
    DOTS1 = Fraction(1, 12)
    DOTS2 = Fraction(1, 6)
    DOTS3 = Fraction(1, 4)
    DOTS4 = Fraction(1, 3)
    DOTS5 = Fraction(5, 12)
    S = Fraction(1, 2)
    SILIQUA = Fraction(1, 1728)
    SCRIPULUM = Fraction(1, 288)
    DIMIDIA_SEXTULA = Fraction(1, 144)
    SEXTULA = Fraction(1, 72)
    SICILICUS = Fraction(1, 48)
    SEMUNCIA = Fraction(1, 24)

    def roman_value(self, _option: Options = DEFAULT_OPTIONS, /) -> Fraction:
        """
        Gets the fractional or integer value this object represents
        """
        return self.value

    def __str__(self) -> str:
        return {
            "DOTS1": "\N{MIDDLE DOT}",
            "DOTS2": "\N{TWO DOT PUNCTUATION}",
            "DOTS3": "\N{THREE DOT PUNCTUATION}",
            "DOTS4": "\N{FOUR DOT PUNCTUATION}",
            "DOTS5": "\N{FIVE DOT PUNCTUATION}",
            "S": "S",
            "SILIQUA": "\N{ROMAN SILIQUA SIGN}",
            "SCRIPULUM": "\N{SCRUPLE}",
            "DIMIDIA_SEXTULA": "\N{ROMAN DIMIDIA SEXTULA SIGN}",
            "SEXTULA": "\N{ROMAN SEXTULA SIGN}",
            "SICILICUS": "\N{ROMAN NUMERAL REVERSED ONE HUNDRED}",
            "SEMUNCIA": "\N{ROMAN SEMUNCIA SIGN}",
        }[self.name]


class ApostrophusRomanSigil(Enum):
    FIVE_HUNDRED = 500
    THOUSAND = 1000
    FIVE_THOUSAND = 5000
    TEN_THOUSAND = 10000
    FIFTY_THOUSAND = 50000
    HUNDRED_THOUSAND = 100000

    def roman_value(self, _options: Options = DEFAULT_OPTIONS, /) -> int:
        """
        Gets the fractional or integer value this object represents
        """
        return self.value

    def __str__(self) -> str:
        return {
            "FIVE_HUNDRED": "D",
            "THOUSAND": "\N{ROMAN NUMERAL ONE THOUSAND C D}",
            "FIVE_THOUSAND": "\N{ROMAN NUMERAL FIVE THOUSAND}",
            "TEN_THOUSAND": "\N{ROMAN NUMERAL TEN THOUSAND}",
            "FIFTY_THOUSAND": "\N{ROMAN NUMERAL FIFTY THOUSAND}",
            "HUNDRED_THOUSAND": "\N{ROMAN NUMERAL ONE HUNDRED THOUSAND}",
        }[self.name]


class VinculumKind(Enum):
    THOUSAND = 1000
    HUNDRED_THOUSAND = 1000000


class Vinculum:
    """
    Represents a vinculum group
    """

    inner: RomanSequence
    kind: VinculumKind
    multiplicity: int

    def __init__(
        self,
        *args: HasRomanValue | str,
        kind: VinculumKind = VinculumKind.THOUSAND,
        multiplicity: int = 1,
    ):
        assert multiplicity >= 1
        sequence = mk_roman_sequence(*args)
        if sequence is None:
            raise ValueError("Cannot parse string as roman numeral")
        self.inner = sequence
        self.kind = kind
        self.multiplicity = multiplicity

    def roman_value(self, options: Options = DEFAULT_OPTIONS, /) -> int | Fraction:
        """
        Gets the fractional or integer value this object represents
        """
        return cast(int, self.kind.value**self.multiplicity) * self.inner.roman_value(
            options
        )

    def __str__(self) -> str:
        if self.kind == VinculumKind.THOUSAND:
            return "".join(
                map(
                    lambda char: char + "\N{COMBINING OVERLINE}" * self.multiplicity,
                    str(self.inner),
                )
            )

        if self.kind == VinculumKind.HUNDRED_THOUSAND:
            inner_str = "".join(
                map(
                    lambda char: char + "\N{COMBINING OVERLINE}" * self.multiplicity,
                    str(self.inner),
                )
            )
            return f"|{inner_str}|"

        raise TypeError("Unreachable")


def count(source: Iterable[object]) -> int:
    return sum(1 for _ in source)


@final
class RomanSequence:
    sequence: Sequence[HasRomanValue]

    def __init__(self, sequence: Sequence[HasRomanValue]):
        self.sequence = sequence

    def __add__(self, other: RomanSequence) -> RomanSequence:
        if not isinstance(other, RomanSequence):
            return NotImplemented
        return RomanSequence([*self.sequence, *other.sequence])

    def flatten(self) -> RomanSequence:
        flattened = RomanSequence([])
        for item in self.sequence:
            if isinstance(item, RomanSequence):
                flattened += item  # .flatten()
            else:
                flattened += RomanSequence([item])
        return flattened

    @staticmethod
    def _group(
        source: Iterable[int | Fraction], do_group: bool = True
    ) -> Iterable[int | Fraction]:
        if not do_group:
            return source
        return (k * count(v) for k, v in groupby(source))

    def _evaluate_with_nested_subtraction_stack(
        self, options: Options = DEFAULT_OPTIONS
    ) -> int | Fraction:
        value_sequence = [
            *self._group(
                (item.roman_value(options) for item in self.sequence), options.grouped
            )
        ]
        summands: list[int | Fraction] = [value_sequence[0]]
        for value in value_sequence[1:]:
            if value > summands[-1]:
                last_summand = summands.pop()
                if options.strict and not (
                    (last_summand == 1 and value in [5, 10])
                    or (last_summand == 10 and value in [50, 100])
                    or (last_summand == 100 and value in [500, 1000])
                ):
                    raise ValueError("Malformed numeral (strict=True)")

                summands.append(value - last_summand)
            else:
                summands.append(value)

        return sum(summands)

    def _evaluate_with_single_value_subtraction(
        self, options: Options = DEFAULT_OPTIONS
    ) -> int | Fraction:
        total: int | Fraction = 0
        last_value: int | Fraction | None = None
        for item in reversed(self.sequence):
            value = item.roman_value(options)
            if last_value is not None and last_value > value:
                if options.strict and not (
                    (value == 1 and last_value in [5, 10])
                    or (value == 10 and last_value in [50, 100])
                    or (value == 100 and last_value in [500, 1000])
                ):
                    raise ValueError("Malformed roman numeral (strict=True)")
                total -= value
                last_value = value
            else:
                total += value
                last_value = value

        return total

    def roman_value(self, options: Options = DEFAULT_OPTIONS, /) -> int | Fraction:
        """
        Gets the fractional or integer value this object represents
        """
        if self.sequence == []:
            raise ValueError("Empty numeral")

        match options.mode:
            case SequenceEvaluationMode.NESTED_SUBTRACTION_STACK_MODE:
                return self._evaluate_with_nested_subtraction_stack(options)

            case SequenceEvaluationMode.SINGLE_VALUE_SUBTRACTION_MODE:
                return self._evaluate_with_single_value_subtraction(options)

    def __str__(self) -> str:
        return "".join(str(item) for item in self.sequence)


I, V, X, L, C, D, M = BasicRomanSigil

(
    DOTS1,
    DOTS2,
    DOTS3,
    DOTS4,
    DOTS5,
    S,
    SILIQUA,
    SCRIPULUM,
    DIMIDIA_SEXTULA,
    SEXTULA,
    SICILICUS,
    SEMUNCIA,
) = FractionalRomanSigil

(
    FIVE_HUNDRED,
    THOUSAND,
    FIVE_THOUSAND,
    TEN_THOUSAND,
    FIFTY_THOUSAND,
    HUNDRED_THOUSAND,
) = ApostrophusRomanSigil

simple_symbol_decompositions: Mapping[str, RomanSequence] = (
    {k: RomanSequence([I]) for k in ("I", "\N{ROMAN NUMERAL ONE}")}
    | {k: RomanSequence([V]) for k in ("V", "\N{ROMAN NUMERAL FIVE}")}
    | {k: RomanSequence([X]) for k in ("X", "\N{ROMAN NUMERAL TEN}")}
    | {k: RomanSequence([L]) for k in ("L", "\N{ROMAN NUMERAL FIFTY}")}
    | {k: RomanSequence([C]) for k in ("C", "\N{ROMAN NUMERAL ONE HUNDRED}")}
    | {k: RomanSequence([D]) for k in ("D", "\N{ROMAN NUMERAL FIVE HUNDRED}")}
    | {k: RomanSequence([M]) for k in ("M", "\N{ROMAN NUMERAL ONE THOUSAND}")}
    | {
        k: RomanSequence([SEMUNCIA])
        for k in (
            "\N{ROMAN SEMUNCIA SIGN}",
            "\N{GREEK CAPITAL LETTER SIGMA}",
            "\N{CYRILLIC CAPITAL LETTER UKRAINIAN IE}",
        )
    }
    | {k: RomanSequence([DOTS1]) for k in ("\N{MIDDLE DOT}", ".")}
    | {k: RomanSequence([DOTS2]) for k in ("\N{TWO DOT PUNCTUATION}", ":")}
    | {
        k: RomanSequence([DOTS3])
        for k in ("\N{THREE DOT PUNCTUATION}", "\N{THEREFORE}")
    }
    | {
        k: RomanSequence([DOTS4])
        for k in ("\N{FOUR DOT PUNCTUATION}", "\N{PROPORTION}")
    }
    | {
        "\N{ROMAN NUMERAL TWO}": RomanSequence([I] * 2),
        "\N{ROMAN NUMERAL THREE}": RomanSequence([I] * 3),
        "\N{ROMAN NUMERAL FOUR}": RomanSequence([I, V]),
        "\N{ROMAN NUMERAL SIX}": RomanSequence([V, I]),
        "\N{ROMAN NUMERAL SEVEN}": RomanSequence([V] + [I] * 2),
        "\N{ROMAN NUMERAL EIGHT}": RomanSequence([V] + [I] * 3),
        "\N{ROMAN NUMERAL NINE}": RomanSequence([I, X]),
        "\N{ROMAN NUMERAL ELEVEN}": RomanSequence([X, I]),
        "\N{ROMAN NUMERAL TWELVE}": RomanSequence([X, I, I]),
        "\N{FIVE DOT PUNCTUATION}": RomanSequence([DOTS5]),
        "S": RomanSequence([S]),
        "\N{ROMAN SILIQUA SIGN}": RomanSequence([SILIQUA]),
        "\N{SCRUPLE}": RomanSequence([SCRIPULUM]),
        "\N{ROMAN DIMIDIA SEXTULA SIGN}": RomanSequence([DIMIDIA_SEXTULA]),
        "\N{ROMAN SEXTULA SIGN}": RomanSequence([SEXTULA]),
        "\N{ROMAN NUMERAL REVERSED ONE HUNDRED}": RomanSequence([SICILICUS]),
        "\N{ROMAN NUMERAL ONE THOUSAND C D}": RomanSequence([THOUSAND]),
        "\N{ROMAN NUMERAL FIVE THOUSAND}": RomanSequence([FIVE_THOUSAND]),
        "\N{ROMAN NUMERAL TEN THOUSAND}": RomanSequence([TEN_THOUSAND]),
        "\N{ROMAN NUMERAL FIFTY THOUSAND}": RomanSequence([FIFTY_THOUSAND]),
        "\N{ROMAN NUMERAL ONE HUNDRED THOUSAND}": RomanSequence([HUNDRED_THOUSAND]),
    }
)


def parse_single_symbol(source: str, begin: int) -> tuple[RomanSequence, int] | None:
    try:
        return (simple_symbol_decompositions[source[begin]], begin + 1)
    except (KeyError, IndexError):
        return None


VINCULUM_OVERLINES = re.compile("\N{COMBINING OVERLINE}+")


def compile_vinculum_overlines(count: int) -> re.Pattern[str]:
    return re.compile("\N{COMBINING OVERLINE}{%d}" % count)


def parse_vinculum_thousand_group(
    source: str, begin: int
) -> tuple[Vinculum, int] | None:
    result = parse_single_symbol(source, begin)
    if result is None:
        return None
    sequence, begin = result
    match = VINCULUM_OVERLINES.match(source, begin)
    if match is None:
        return None
    begin = match.end()
    count = len(match.group(0))
    overlines = compile_vinculum_overlines(count)
    while True:
        result = parse_single_symbol(source, begin)
        if result is None:
            break
        match = overlines.match(source, result[1])
        if match is None:
            break
        sequence += result[0]
        begin = match.end()

    return (Vinculum(sequence, kind=VinculumKind.THOUSAND, multiplicity=count), begin)


def parse_vinculum_hundred_thousand_group(
    source: str, begin: int
) -> tuple[Vinculum, int] | None:
    try:
        if source[begin] != "|":
            return None
    except IndexError:
        return None

    result = parse_vinculum_thousand_group(source, begin + 1)
    if result is None:
        return None

    component, begin = result

    try:
        if source[begin] != "|":
            return None
    except IndexError:
        return None

    return (Vinculum(component.inner, kind=VinculumKind.HUNDRED_THOUSAND), begin + 1)


def parse_roman_sequence(source: str, begin: int) -> tuple[RomanSequence, int] | None:
    sequence: list[HasRomanValue] = []
    while True:
        result = parse_roman_component(source, begin, False)
        if result is None:
            break

        sequence.append(result[0])
        begin = result[1]

    if len(sequence) == 0:
        return None

    return (RomanSequence(sequence), begin)


APOSTROPHUS = re.compile(
    "([C\N{ROMAN NUMERAL ONE HUNDRED}]*)[I\N{ROMAN NUMERAL ONE}]([\N{ROMAN NUMERAL REVERSED ONE HUNDRED})]+)"
)

apostrophus_table: dict[tuple[int, int], ApostrophusRomanSigil] = {
    (0, 1): ApostrophusRomanSigil.FIVE_HUNDRED,
    (0, 2): ApostrophusRomanSigil.FIVE_THOUSAND,
    (0, 3): ApostrophusRomanSigil.FIFTY_THOUSAND,
    (1, 1): ApostrophusRomanSigil.THOUSAND,
    (2, 2): ApostrophusRomanSigil.TEN_THOUSAND,
    (3, 3): ApostrophusRomanSigil.HUNDRED_THOUSAND,
}


def parse_apostrophus(
    source: str, begin: int
) -> tuple[ApostrophusRomanSigil, int] | None:
    match = APOSTROPHUS.match(source, begin)
    if match is None:
        return None

    begin = match.end()
    left_count = len(match.group(1))
    right_count = len(match.group(2))
    try:
        return (apostrophus_table[(left_count, right_count)], begin)
    except KeyError:
        return None


def parse_roman_component(
    source: str, begin: int, sequence: bool = True
) -> tuple[HasRomanValue, int] | None:
    result: tuple[HasRomanValue, int] | None
    result = parse_vinculum_thousand_group(source, begin)
    if result is not None:
        return result

    result = parse_vinculum_hundred_thousand_group(source, begin)
    if result is not None:
        return result

    result = parse_apostrophus(source, begin)
    if result is not None:
        return result

    if sequence:
        result = parse_roman_sequence(source, begin)
        if result is not None:
            return result

    result = parse_single_symbol(source, begin)
    if result is not None:
        return result

    return None


def mk_roman_sequence(*args: HasRomanValue | str) -> RomanSequence:
    """
    Create an object representing a sequence of roman symbols
    """
    sequence: list[HasRomanValue] = []
    if len(args) == 0:
        raise ValueError("Empty input roman numeral")

    for arg in args:
        if isinstance(arg, str):
            result = parse_roman_sequence("".join(arg.split()).upper(), 0)
            if result is None:
                raise ValueError("Cannot parse string as roman numeral")

            sequence.append(result[0])
        else:
            sequence.append(arg)

    return RomanSequence(sequence).flatten()


def get_roman_value(
    *args: HasRomanValue | str, options: Options = DEFAULT_OPTIONS
) -> int | Fraction:
    """
    Gets the fractional or integer value this roman sequence represents
    """
    sequence = mk_roman_sequence(*args)
    return sequence.roman_value(options)
