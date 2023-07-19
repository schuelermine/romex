# Romex

Romex is a roman-numeral-to-python-number converter that supports many kinds of roman numerals.

Romex supports:
- Modern roman numerals (IV, XI, MMXXIII)
- Other subtractive forms (IXL)
- Vinculum (either as a Python object or Unicode overlines)
- Apostrophus (either with a reversed C or Unicode symbols)
- Fractional numerals (either as dots or Unicode symbols)
- Strict evaluation mode that rejects numerals that don't fit modern convention

## API

Roman symbols:
- Simple symbols: I, V, X, L, C, D, M
- Fractional symbols: DOTS1, DOTS2, DOTS3, DOTS4, DOTS5, S, SILIQUA, SCRIPULUM, DIMIDIA_SEXTULA, SICILICUS, SEMUNCIA
- Apostrophus: FIVE_HUNDRED, THOUSAND, FIVE_THOUSAND, TEN_THOUSAND, FIFTY_THOUSAND, HUNDRED_THOUSAND

Vinculum:
- Vinculum
- VinculumKind

Sequence parser and evaluator:
- mk_roman_sequence
- get_roman_value
