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

parser grammar SQLParser;

options { tokenVocab = SQLLexer; }

logic_expr_unit
    : logic_expr EOF
    ;

logic_expr
    : or_expr
    ;

or_expr
    : and_expr (OR and_expr)*
    ;

and_expr
    : primary_expr (AND primary_expr)*
    ;

primary_expr
    : relation_expr
    | LP logic_expr RP
    ;

relation_expr
    : identifier rel_oper value_expr
    | identifier LIKE value_expr
    | identifier NOT? IN LP in_value_expr_list RP
    | identifier NOT? (CONTAIN_ALL | CONTAIN_ANY) LP in_value_expr_list? RP
    | identifier IS NOT? NULL_V
    | function_call rel_oper value_expr
    ;

rel_oper
    : E_OP
    | NE_OP
    | L_OP
    | G_OP
    | L_OP E_OP
    | G_OP E_OP
    | LE_OP
    | GE_OP
    ;

value_expr
    : constant
    | function_call
    ;

in_value_expr_list
    : in_value_expr (COMMA in_value_expr)*
    ;

in_value_expr
    : numeric
    | quoted_string
    | bool_value
    ;

constant
    : numeric
    | quoted_string
    | vector_expr
    | bool_value
    ;

vector_expr
    : vector
    | LMP vector (COMMA vector)* RMP
    ;

vector
    : LMP numeric (COMMA numeric)* RMP
    ;

function_value_expr
    : value_expr
    | identifier
    ;

function_call
    : identifier LP (function_value_expr (COMMA function_value_expr)*)? RP
    ;

numeric
    : INTEGER
    | FLOAT
    ;

quoted_string
    : SQUOTA_STRING
    | DQUOTA_STRING
    ;

bool_value
    : TRUE_V
    | FALSE_V
    ;

identifier
    : REGULAR_ID
    | OR
    | AND
    | NOT
    | IN
    | BETWEEN
    | LIKE
    | WHERE
    | SELECT
    | AS
    | BY
    | ORDER
    | ASC
    | DESC
    | LIMIT
    ;
