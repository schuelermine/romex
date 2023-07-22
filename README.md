# Romex

Romex is a roman-numeral-to-python-integer-or-fraction converter that supports many kinds of roman numerals.

Romex supports:
- Modern roman numerals (IV, XI, MMXXIII)
- Other subtractive forms (IXL)
- Vinculum (either as a Python object or Unicode overlines)
- Apostrophus (either with a reversed C or Unicode symbols)
- Fractional numerals (either as periods or Unicode symbols)
- Strict sequence evaluation flag that rejects numerals that don't fit modern convention
- Three sequence evaluation modes for non-strict parsing

## API

Roman symbols:
- Simple symbols: `I`, `V`, `X`, `L`, `C`, `D`, `M`
- Fractional symbols: `DOTS1`, `DOTS2`, `DOTS3`, `DOTS4`, `DOTS5`, `S`, `SILIQUA`, `SCRIPULUM`, `DIMIDIA_SEXTULA`, `SICILICUS`, `SEMUNCIA`
- Apostrophus: `FIVE_HUNDRED`, `THOUSAND`, `FIVE_THOUSAND`, `TEN_THOUSAND`, `FIFTY_THOUSAND`, `HUNDRED_THOUSAND`

Vinculum:
- `Vinculum(*args, kind=..., multiplicity=...)`: Construct by passing any sequence of roman symbols or vinculum, and a kind and multiplicity
- `VinculumKind` (an enum)
    - `THOUSAND`
    - `HUNDRED_THOUSAND`

Sequence parser and evaluator:
- `mk_roman_sequence(*args)`: Takes any sequence of roman symbols or vinculum and produces an object representing them
- `get_roman_value(*args, options=...)`: Takes any sequence of roman symbols or vinculum as well as options and returns the fractional or integer value of this sequence

Common method for roman symbol, vinculum, and `mk_roman_sequence(...)`:
- `roman_value(options)`: Gets the fractional or integer value this object represents

Options:
- `Options(mode=..., *, strict=False, grouped=True)`: Represents the mode for evaluating sequences
    - `strict`: Disallow sequences with non-modern subtractive forms
    - `grouped`: Group some identical values before potentially building a subtractive form across them
      Note that Vinculum always constitutes a group, even with `grouped=False`
- `NESTED_SUBTRACTION_STACK_MODE`: Any subtractive sequence can act as the subtracting part of a subtractive form
- `SINGLE_VALUE_SUBTRACTION_MODE`: Only single values can act as the subtracting part of a subtractive form
  Note that Vinculum always constitutes a single value
