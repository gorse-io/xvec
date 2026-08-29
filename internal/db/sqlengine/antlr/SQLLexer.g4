// Copyright 2026-present the xvec project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

lexer grammar SQLLexer;

options { caseInsensitive = true; }

channels { COMMENTS }

OR: 'OR';
AND: 'AND';
NOT: 'NOT';
IN: 'IN';
CONTAIN_ALL: 'CONTAIN_ALL';
CONTAIN_ANY: 'CONTAIN_ANY';
BETWEEN: 'BETWEEN';
LIKE: 'LIKE';
WHERE: 'WHERE';
SELECT: 'SELECT';
FROM: 'FROM';
AS: 'AS';
BY: 'BY';
ORDER: 'ORDER';
ASC: 'ASC';
DESC: 'DESC';
LIMIT: 'LIMIT';
TRUE_V: 'TRUE';
FALSE_V: 'FALSE';
IS: 'IS';
NULL_V: 'NULL';

fragment UNSIGNED_INTEGER: UNSIGNED_INTEGER_FRAGMENT;
INTEGER: MINUS_SIGN? UNSIGNED_INTEGER;

fragment APPROXIMATE_NUM_LIT
    : FLOAT_FRAGMENT ('E' ('+' | '-')? (FLOAT_FRAGMENT | UNSIGNED_INTEGER_FRAGMENT))? ('D' | 'F')?
    ;
FLOAT: MINUS_SIGN? APPROXIMATE_NUM_LIT;

SQUOTA_STRING: '\'' (~('\'' | '\\') | '\\'.)* '\'';
DQUOTA_STRING: '"' (~('"' | '\\') | '\\'.)* '"';

LP: '(';
RP: ')';
LMP: '[';
RMP: ']';
COMMA: ',';
LE_OP: '<=';
GE_OP: '>=';
NE_OP: '!=';
L_OP: '<';
G_OP: '>';
E_OP: '=';
MINUS_SIGN: '-';
UNDERSCORE: '_';

SPACES: [ \t\r\n]+ -> skip;
SINGLE_LINE_COMMENT: '--' ~('\r' | '\n')* ('\r'? '\n' | EOF) -> channel(COMMENTS);
MULTI_LINE_COMMENT: '/*' .*? '*/' -> channel(COMMENTS);

fragment SIMPLE_LETTER: [A-Z];
fragment UNSIGNED_INTEGER_FRAGMENT: [0-9]+;
fragment FLOAT_FRAGMENT: UNSIGNED_INTEGER* '.'? UNSIGNED_INTEGER+;

REGULAR_ID: (SIMPLE_LETTER | '_' | '-' | [0-9])+;
