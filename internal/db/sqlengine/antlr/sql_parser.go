// Code generated from SQLParser.g4 by ANTLR 4.13.1. DO NOT EDIT.

package antlr // SQLParser
import (
	"fmt"
	"strconv"
	"sync"

	"github.com/antlr4-go/antlr/v4"
)

// Suppress unused import errors
var _ = fmt.Printf
var _ = strconv.Itoa
var _ = sync.Once{}

type SQLParser struct {
	*antlr.BaseParser
}

var SQLParserParserStaticData struct {
	once                   sync.Once
	serializedATN          []int32
	LiteralNames           []string
	SymbolicNames          []string
	RuleNames              []string
	PredictionContextCache *antlr.PredictionContextCache
	atn                    *antlr.ATN
	decisionToDFA          []*antlr.DFA
}

func sqlparserParserInit() {
	staticData := &SQLParserParserStaticData
	staticData.LiteralNames = []string{
		"", "'OR'", "'AND'", "'NOT'", "'IN'", "'CONTAIN_ALL'", "'CONTAIN_ANY'",
		"'BETWEEN'", "'LIKE'", "'WHERE'", "'SELECT'", "'FROM'", "'AS'", "'BY'",
		"'ORDER'", "'ASC'", "'DESC'", "'LIMIT'", "'TRUE'", "'FALSE'", "'IS'",
		"'NULL'", "", "", "", "", "'('", "')'", "'['", "']'", "','", "'<='",
		"'>='", "'!='", "'<'", "'>'", "'='", "'-'", "'_'",
	}
	staticData.SymbolicNames = []string{
		"", "OR", "AND", "NOT", "IN", "CONTAIN_ALL", "CONTAIN_ANY", "BETWEEN",
		"LIKE", "WHERE", "SELECT", "FROM", "AS", "BY", "ORDER", "ASC", "DESC",
		"LIMIT", "TRUE_V", "FALSE_V", "IS", "NULL_V", "INTEGER", "FLOAT", "SQUOTA_STRING",
		"DQUOTA_STRING", "LP", "RP", "LMP", "RMP", "COMMA", "LE_OP", "GE_OP",
		"NE_OP", "L_OP", "G_OP", "E_OP", "MINUS_SIGN", "UNDERSCORE", "SPACES",
		"SINGLE_LINE_COMMENT", "MULTI_LINE_COMMENT", "REGULAR_ID",
	}
	staticData.RuleNames = []string{
		"logic_expr_unit", "logic_expr", "or_expr", "and_expr", "primary_expr",
		"relation_expr", "rel_oper", "value_expr", "in_value_expr_list", "in_value_expr",
		"constant", "vector_expr", "vector", "function_value_expr", "function_call",
		"numeric", "quoted_string", "bool_value", "identifier",
	}
	staticData.PredictionContextCache = antlr.NewPredictionContextCache()
	staticData.serializedATN = []int32{
		4, 1, 42, 194, 2, 0, 7, 0, 2, 1, 7, 1, 2, 2, 7, 2, 2, 3, 7, 3, 2, 4, 7,
		4, 2, 5, 7, 5, 2, 6, 7, 6, 2, 7, 7, 7, 2, 8, 7, 8, 2, 9, 7, 9, 2, 10, 7,
		10, 2, 11, 7, 11, 2, 12, 7, 12, 2, 13, 7, 13, 2, 14, 7, 14, 2, 15, 7, 15,
		2, 16, 7, 16, 2, 17, 7, 17, 2, 18, 7, 18, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1,
		1, 2, 1, 2, 1, 2, 5, 2, 47, 8, 2, 10, 2, 12, 2, 50, 9, 2, 1, 3, 1, 3, 1,
		3, 5, 3, 55, 8, 3, 10, 3, 12, 3, 58, 9, 3, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4,
		3, 4, 65, 8, 4, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1,
		5, 3, 5, 77, 8, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 3, 5, 86,
		8, 5, 1, 5, 1, 5, 1, 5, 3, 5, 91, 8, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 3,
		5, 98, 8, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 1, 5, 3, 5, 106, 8, 5, 1, 6,
		1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 1, 6, 3, 6, 118, 8, 6,
		1, 7, 1, 7, 3, 7, 122, 8, 7, 1, 8, 1, 8, 1, 8, 5, 8, 127, 8, 8, 10, 8,
		12, 8, 130, 9, 8, 1, 9, 1, 9, 1, 9, 3, 9, 135, 8, 9, 1, 10, 1, 10, 1, 10,
		1, 10, 3, 10, 141, 8, 10, 1, 11, 1, 11, 1, 11, 1, 11, 1, 11, 5, 11, 148,
		8, 11, 10, 11, 12, 11, 151, 9, 11, 1, 11, 1, 11, 3, 11, 155, 8, 11, 1,
		12, 1, 12, 1, 12, 1, 12, 5, 12, 161, 8, 12, 10, 12, 12, 12, 164, 9, 12,
		1, 12, 1, 12, 1, 13, 1, 13, 3, 13, 170, 8, 13, 1, 14, 1, 14, 1, 14, 1,
		14, 1, 14, 5, 14, 177, 8, 14, 10, 14, 12, 14, 180, 9, 14, 3, 14, 182, 8,
		14, 1, 14, 1, 14, 1, 15, 1, 15, 1, 16, 1, 16, 1, 17, 1, 17, 1, 18, 1, 18,
		1, 18, 0, 0, 19, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28,
		30, 32, 34, 36, 0, 5, 1, 0, 5, 6, 1, 0, 22, 23, 1, 0, 24, 25, 1, 0, 18,
		19, 4, 0, 1, 4, 7, 10, 12, 17, 42, 42, 206, 0, 38, 1, 0, 0, 0, 2, 41, 1,
		0, 0, 0, 4, 43, 1, 0, 0, 0, 6, 51, 1, 0, 0, 0, 8, 64, 1, 0, 0, 0, 10, 105,
		1, 0, 0, 0, 12, 117, 1, 0, 0, 0, 14, 121, 1, 0, 0, 0, 16, 123, 1, 0, 0,
		0, 18, 134, 1, 0, 0, 0, 20, 140, 1, 0, 0, 0, 22, 154, 1, 0, 0, 0, 24, 156,
		1, 0, 0, 0, 26, 169, 1, 0, 0, 0, 28, 171, 1, 0, 0, 0, 30, 185, 1, 0, 0,
		0, 32, 187, 1, 0, 0, 0, 34, 189, 1, 0, 0, 0, 36, 191, 1, 0, 0, 0, 38, 39,
		3, 2, 1, 0, 39, 40, 5, 0, 0, 1, 40, 1, 1, 0, 0, 0, 41, 42, 3, 4, 2, 0,
		42, 3, 1, 0, 0, 0, 43, 48, 3, 6, 3, 0, 44, 45, 5, 1, 0, 0, 45, 47, 3, 6,
		3, 0, 46, 44, 1, 0, 0, 0, 47, 50, 1, 0, 0, 0, 48, 46, 1, 0, 0, 0, 48, 49,
		1, 0, 0, 0, 49, 5, 1, 0, 0, 0, 50, 48, 1, 0, 0, 0, 51, 56, 3, 8, 4, 0,
		52, 53, 5, 2, 0, 0, 53, 55, 3, 8, 4, 0, 54, 52, 1, 0, 0, 0, 55, 58, 1,
		0, 0, 0, 56, 54, 1, 0, 0, 0, 56, 57, 1, 0, 0, 0, 57, 7, 1, 0, 0, 0, 58,
		56, 1, 0, 0, 0, 59, 65, 3, 10, 5, 0, 60, 61, 5, 26, 0, 0, 61, 62, 3, 2,
		1, 0, 62, 63, 5, 27, 0, 0, 63, 65, 1, 0, 0, 0, 64, 59, 1, 0, 0, 0, 64,
		60, 1, 0, 0, 0, 65, 9, 1, 0, 0, 0, 66, 67, 3, 36, 18, 0, 67, 68, 3, 12,
		6, 0, 68, 69, 3, 14, 7, 0, 69, 106, 1, 0, 0, 0, 70, 71, 3, 36, 18, 0, 71,
		72, 5, 8, 0, 0, 72, 73, 3, 14, 7, 0, 73, 106, 1, 0, 0, 0, 74, 76, 3, 36,
		18, 0, 75, 77, 5, 3, 0, 0, 76, 75, 1, 0, 0, 0, 76, 77, 1, 0, 0, 0, 77,
		78, 1, 0, 0, 0, 78, 79, 5, 4, 0, 0, 79, 80, 5, 26, 0, 0, 80, 81, 3, 16,
		8, 0, 81, 82, 5, 27, 0, 0, 82, 106, 1, 0, 0, 0, 83, 85, 3, 36, 18, 0, 84,
		86, 5, 3, 0, 0, 85, 84, 1, 0, 0, 0, 85, 86, 1, 0, 0, 0, 86, 87, 1, 0, 0,
		0, 87, 88, 7, 0, 0, 0, 88, 90, 5, 26, 0, 0, 89, 91, 3, 16, 8, 0, 90, 89,
		1, 0, 0, 0, 90, 91, 1, 0, 0, 0, 91, 92, 1, 0, 0, 0, 92, 93, 5, 27, 0, 0,
		93, 106, 1, 0, 0, 0, 94, 95, 3, 36, 18, 0, 95, 97, 5, 20, 0, 0, 96, 98,
		5, 3, 0, 0, 97, 96, 1, 0, 0, 0, 97, 98, 1, 0, 0, 0, 98, 99, 1, 0, 0, 0,
		99, 100, 5, 21, 0, 0, 100, 106, 1, 0, 0, 0, 101, 102, 3, 28, 14, 0, 102,
		103, 3, 12, 6, 0, 103, 104, 3, 14, 7, 0, 104, 106, 1, 0, 0, 0, 105, 66,
		1, 0, 0, 0, 105, 70, 1, 0, 0, 0, 105, 74, 1, 0, 0, 0, 105, 83, 1, 0, 0,
		0, 105, 94, 1, 0, 0, 0, 105, 101, 1, 0, 0, 0, 106, 11, 1, 0, 0, 0, 107,
		118, 5, 36, 0, 0, 108, 118, 5, 33, 0, 0, 109, 118, 5, 34, 0, 0, 110, 118,
		5, 35, 0, 0, 111, 112, 5, 34, 0, 0, 112, 118, 5, 36, 0, 0, 113, 114, 5,
		35, 0, 0, 114, 118, 5, 36, 0, 0, 115, 118, 5, 31, 0, 0, 116, 118, 5, 32,
		0, 0, 117, 107, 1, 0, 0, 0, 117, 108, 1, 0, 0, 0, 117, 109, 1, 0, 0, 0,
		117, 110, 1, 0, 0, 0, 117, 111, 1, 0, 0, 0, 117, 113, 1, 0, 0, 0, 117,
		115, 1, 0, 0, 0, 117, 116, 1, 0, 0, 0, 118, 13, 1, 0, 0, 0, 119, 122, 3,
		20, 10, 0, 120, 122, 3, 28, 14, 0, 121, 119, 1, 0, 0, 0, 121, 120, 1, 0,
		0, 0, 122, 15, 1, 0, 0, 0, 123, 128, 3, 18, 9, 0, 124, 125, 5, 30, 0, 0,
		125, 127, 3, 18, 9, 0, 126, 124, 1, 0, 0, 0, 127, 130, 1, 0, 0, 0, 128,
		126, 1, 0, 0, 0, 128, 129, 1, 0, 0, 0, 129, 17, 1, 0, 0, 0, 130, 128, 1,
		0, 0, 0, 131, 135, 3, 30, 15, 0, 132, 135, 3, 32, 16, 0, 133, 135, 3, 34,
		17, 0, 134, 131, 1, 0, 0, 0, 134, 132, 1, 0, 0, 0, 134, 133, 1, 0, 0, 0,
		135, 19, 1, 0, 0, 0, 136, 141, 3, 30, 15, 0, 137, 141, 3, 32, 16, 0, 138,
		141, 3, 22, 11, 0, 139, 141, 3, 34, 17, 0, 140, 136, 1, 0, 0, 0, 140, 137,
		1, 0, 0, 0, 140, 138, 1, 0, 0, 0, 140, 139, 1, 0, 0, 0, 141, 21, 1, 0,
		0, 0, 142, 155, 3, 24, 12, 0, 143, 144, 5, 28, 0, 0, 144, 149, 3, 24, 12,
		0, 145, 146, 5, 30, 0, 0, 146, 148, 3, 24, 12, 0, 147, 145, 1, 0, 0, 0,
		148, 151, 1, 0, 0, 0, 149, 147, 1, 0, 0, 0, 149, 150, 1, 0, 0, 0, 150,
		152, 1, 0, 0, 0, 151, 149, 1, 0, 0, 0, 152, 153, 5, 29, 0, 0, 153, 155,
		1, 0, 0, 0, 154, 142, 1, 0, 0, 0, 154, 143, 1, 0, 0, 0, 155, 23, 1, 0,
		0, 0, 156, 157, 5, 28, 0, 0, 157, 162, 3, 30, 15, 0, 158, 159, 5, 30, 0,
		0, 159, 161, 3, 30, 15, 0, 160, 158, 1, 0, 0, 0, 161, 164, 1, 0, 0, 0,
		162, 160, 1, 0, 0, 0, 162, 163, 1, 0, 0, 0, 163, 165, 1, 0, 0, 0, 164,
		162, 1, 0, 0, 0, 165, 166, 5, 29, 0, 0, 166, 25, 1, 0, 0, 0, 167, 170,
		3, 14, 7, 0, 168, 170, 3, 36, 18, 0, 169, 167, 1, 0, 0, 0, 169, 168, 1,
		0, 0, 0, 170, 27, 1, 0, 0, 0, 171, 172, 3, 36, 18, 0, 172, 181, 5, 26,
		0, 0, 173, 178, 3, 26, 13, 0, 174, 175, 5, 30, 0, 0, 175, 177, 3, 26, 13,
		0, 176, 174, 1, 0, 0, 0, 177, 180, 1, 0, 0, 0, 178, 176, 1, 0, 0, 0, 178,
		179, 1, 0, 0, 0, 179, 182, 1, 0, 0, 0, 180, 178, 1, 0, 0, 0, 181, 173,
		1, 0, 0, 0, 181, 182, 1, 0, 0, 0, 182, 183, 1, 0, 0, 0, 183, 184, 5, 27,
		0, 0, 184, 29, 1, 0, 0, 0, 185, 186, 7, 1, 0, 0, 186, 31, 1, 0, 0, 0, 187,
		188, 7, 2, 0, 0, 188, 33, 1, 0, 0, 0, 189, 190, 7, 3, 0, 0, 190, 35, 1,
		0, 0, 0, 191, 192, 7, 4, 0, 0, 192, 37, 1, 0, 0, 0, 19, 48, 56, 64, 76,
		85, 90, 97, 105, 117, 121, 128, 134, 140, 149, 154, 162, 169, 178, 181,
	}
	deserializer := antlr.NewATNDeserializer(nil)
	staticData.atn = deserializer.Deserialize(staticData.serializedATN)
	atn := staticData.atn
	staticData.decisionToDFA = make([]*antlr.DFA, len(atn.DecisionToState))
	decisionToDFA := staticData.decisionToDFA
	for index, state := range atn.DecisionToState {
		decisionToDFA[index] = antlr.NewDFA(state, index)
	}
}

// SQLParserInit initializes any static state used to implement SQLParser. By default the
// static state used to implement the parser is lazily initialized during the first call to
// NewSQLParser(). You can call this function if you wish to initialize the static state ahead
// of time.
func SQLParserInit() {
	staticData := &SQLParserParserStaticData
	staticData.once.Do(sqlparserParserInit)
}

// NewSQLParser produces a new parser instance for the optional input antlr.TokenStream.
func NewSQLParser(input antlr.TokenStream) *SQLParser {
	SQLParserInit()
	this := new(SQLParser)
	this.BaseParser = antlr.NewBaseParser(input)
	staticData := &SQLParserParserStaticData
	this.Interpreter = antlr.NewParserATNSimulator(this, staticData.atn, staticData.decisionToDFA, staticData.PredictionContextCache)
	this.RuleNames = staticData.RuleNames
	this.LiteralNames = staticData.LiteralNames
	this.SymbolicNames = staticData.SymbolicNames
	this.GrammarFileName = "SQLParser.g4"

	return this
}

// SQLParser tokens.
const (
	SQLParserEOF                 = antlr.TokenEOF
	SQLParserOR                  = 1
	SQLParserAND                 = 2
	SQLParserNOT                 = 3
	SQLParserIN                  = 4
	SQLParserCONTAIN_ALL         = 5
	SQLParserCONTAIN_ANY         = 6
	SQLParserBETWEEN             = 7
	SQLParserLIKE                = 8
	SQLParserWHERE               = 9
	SQLParserSELECT              = 10
	SQLParserFROM                = 11
	SQLParserAS                  = 12
	SQLParserBY                  = 13
	SQLParserORDER               = 14
	SQLParserASC                 = 15
	SQLParserDESC                = 16
	SQLParserLIMIT               = 17
	SQLParserTRUE_V              = 18
	SQLParserFALSE_V             = 19
	SQLParserIS                  = 20
	SQLParserNULL_V              = 21
	SQLParserINTEGER             = 22
	SQLParserFLOAT               = 23
	SQLParserSQUOTA_STRING       = 24
	SQLParserDQUOTA_STRING       = 25
	SQLParserLP                  = 26
	SQLParserRP                  = 27
	SQLParserLMP                 = 28
	SQLParserRMP                 = 29
	SQLParserCOMMA               = 30
	SQLParserLE_OP               = 31
	SQLParserGE_OP               = 32
	SQLParserNE_OP               = 33
	SQLParserL_OP                = 34
	SQLParserG_OP                = 35
	SQLParserE_OP                = 36
	SQLParserMINUS_SIGN          = 37
	SQLParserUNDERSCORE          = 38
	SQLParserSPACES              = 39
	SQLParserSINGLE_LINE_COMMENT = 40
	SQLParserMULTI_LINE_COMMENT  = 41
	SQLParserREGULAR_ID          = 42
)

// SQLParser rules.
const (
	SQLParserRULE_logic_expr_unit     = 0
	SQLParserRULE_logic_expr          = 1
	SQLParserRULE_or_expr             = 2
	SQLParserRULE_and_expr            = 3
	SQLParserRULE_primary_expr        = 4
	SQLParserRULE_relation_expr       = 5
	SQLParserRULE_rel_oper            = 6
	SQLParserRULE_value_expr          = 7
	SQLParserRULE_in_value_expr_list  = 8
	SQLParserRULE_in_value_expr       = 9
	SQLParserRULE_constant            = 10
	SQLParserRULE_vector_expr         = 11
	SQLParserRULE_vector              = 12
	SQLParserRULE_function_value_expr = 13
	SQLParserRULE_function_call       = 14
	SQLParserRULE_numeric             = 15
	SQLParserRULE_quoted_string       = 16
	SQLParserRULE_bool_value          = 17
	SQLParserRULE_identifier          = 18
)

// ILogic_expr_unitContext is an interface to support dynamic dispatch.
type ILogic_expr_unitContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Logic_expr() ILogic_exprContext
	EOF() antlr.TerminalNode

	// IsLogic_expr_unitContext differentiates from other interfaces.
	IsLogic_expr_unitContext()
}

type Logic_expr_unitContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyLogic_expr_unitContext() *Logic_expr_unitContext {
	var p = new(Logic_expr_unitContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_logic_expr_unit
	return p
}

func InitEmptyLogic_expr_unitContext(p *Logic_expr_unitContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_logic_expr_unit
}

func (*Logic_expr_unitContext) IsLogic_expr_unitContext() {}

func NewLogic_expr_unitContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Logic_expr_unitContext {
	var p = new(Logic_expr_unitContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_logic_expr_unit

	return p
}

func (s *Logic_expr_unitContext) GetParser() antlr.Parser { return s.parser }

func (s *Logic_expr_unitContext) Logic_expr() ILogic_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(ILogic_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(ILogic_exprContext)
}

func (s *Logic_expr_unitContext) EOF() antlr.TerminalNode {
	return s.GetToken(SQLParserEOF, 0)
}

func (s *Logic_expr_unitContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Logic_expr_unitContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Logic_expr_unit() (localctx ILogic_expr_unitContext) {
	localctx = NewLogic_expr_unitContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 0, SQLParserRULE_logic_expr_unit)
	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(38)
		p.Logic_expr()
	}
	{
		p.SetState(39)
		p.Match(SQLParserEOF)
		if p.HasError() {
			// Recognition error - abort rule
			goto errorExit
		}
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// ILogic_exprContext is an interface to support dynamic dispatch.
type ILogic_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Or_expr() IOr_exprContext

	// IsLogic_exprContext differentiates from other interfaces.
	IsLogic_exprContext()
}

type Logic_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyLogic_exprContext() *Logic_exprContext {
	var p = new(Logic_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_logic_expr
	return p
}

func InitEmptyLogic_exprContext(p *Logic_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_logic_expr
}

func (*Logic_exprContext) IsLogic_exprContext() {}

func NewLogic_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Logic_exprContext {
	var p = new(Logic_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_logic_expr

	return p
}

func (s *Logic_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Logic_exprContext) Or_expr() IOr_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IOr_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IOr_exprContext)
}

func (s *Logic_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Logic_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Logic_expr() (localctx ILogic_exprContext) {
	localctx = NewLogic_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 2, SQLParserRULE_logic_expr)
	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(41)
		p.Or_expr()
	}

	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IOr_exprContext is an interface to support dynamic dispatch.
type IOr_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	AllAnd_expr() []IAnd_exprContext
	And_expr(i int) IAnd_exprContext
	AllOR() []antlr.TerminalNode
	OR(i int) antlr.TerminalNode

	// IsOr_exprContext differentiates from other interfaces.
	IsOr_exprContext()
}

type Or_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyOr_exprContext() *Or_exprContext {
	var p = new(Or_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_or_expr
	return p
}

func InitEmptyOr_exprContext(p *Or_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_or_expr
}

func (*Or_exprContext) IsOr_exprContext() {}

func NewOr_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Or_exprContext {
	var p = new(Or_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_or_expr

	return p
}

func (s *Or_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Or_exprContext) AllAnd_expr() []IAnd_exprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IAnd_exprContext); ok {
			len++
		}
	}

	tst := make([]IAnd_exprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IAnd_exprContext); ok {
			tst[i] = t.(IAnd_exprContext)
			i++
		}
	}

	return tst
}

func (s *Or_exprContext) And_expr(i int) IAnd_exprContext {
	var t antlr.RuleContext
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IAnd_exprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext)
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IAnd_exprContext)
}

func (s *Or_exprContext) AllOR() []antlr.TerminalNode {
	return s.GetTokens(SQLParserOR)
}

func (s *Or_exprContext) OR(i int) antlr.TerminalNode {
	return s.GetToken(SQLParserOR, i)
}

func (s *Or_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Or_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Or_expr() (localctx IOr_exprContext) {
	localctx = NewOr_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 4, SQLParserRULE_or_expr)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(43)
		p.And_expr()
	}
	p.SetState(48)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)

	for _la == SQLParserOR {
		{
			p.SetState(44)
			p.Match(SQLParserOR)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(45)
			p.And_expr()
		}

		p.SetState(50)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IAnd_exprContext is an interface to support dynamic dispatch.
type IAnd_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	AllPrimary_expr() []IPrimary_exprContext
	Primary_expr(i int) IPrimary_exprContext
	AllAND() []antlr.TerminalNode
	AND(i int) antlr.TerminalNode

	// IsAnd_exprContext differentiates from other interfaces.
	IsAnd_exprContext()
}

type And_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyAnd_exprContext() *And_exprContext {
	var p = new(And_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_and_expr
	return p
}

func InitEmptyAnd_exprContext(p *And_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_and_expr
}

func (*And_exprContext) IsAnd_exprContext() {}

func NewAnd_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *And_exprContext {
	var p = new(And_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_and_expr

	return p
}

func (s *And_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *And_exprContext) AllPrimary_expr() []IPrimary_exprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IPrimary_exprContext); ok {
			len++
		}
	}

	tst := make([]IPrimary_exprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IPrimary_exprContext); ok {
			tst[i] = t.(IPrimary_exprContext)
			i++
		}
	}

	return tst
}

func (s *And_exprContext) Primary_expr(i int) IPrimary_exprContext {
	var t antlr.RuleContext
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IPrimary_exprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext)
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IPrimary_exprContext)
}

func (s *And_exprContext) AllAND() []antlr.TerminalNode {
	return s.GetTokens(SQLParserAND)
}

func (s *And_exprContext) AND(i int) antlr.TerminalNode {
	return s.GetToken(SQLParserAND, i)
}

func (s *And_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *And_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) And_expr() (localctx IAnd_exprContext) {
	localctx = NewAnd_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 6, SQLParserRULE_and_expr)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(51)
		p.Primary_expr()
	}
	p.SetState(56)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)

	for _la == SQLParserAND {
		{
			p.SetState(52)
			p.Match(SQLParserAND)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(53)
			p.Primary_expr()
		}

		p.SetState(58)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IPrimary_exprContext is an interface to support dynamic dispatch.
type IPrimary_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Relation_expr() IRelation_exprContext
	LP() antlr.TerminalNode
	Logic_expr() ILogic_exprContext
	RP() antlr.TerminalNode

	// IsPrimary_exprContext differentiates from other interfaces.
	IsPrimary_exprContext()
}

type Primary_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyPrimary_exprContext() *Primary_exprContext {
	var p = new(Primary_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_primary_expr
	return p
}

func InitEmptyPrimary_exprContext(p *Primary_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_primary_expr
}

func (*Primary_exprContext) IsPrimary_exprContext() {}

func NewPrimary_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Primary_exprContext {
	var p = new(Primary_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_primary_expr

	return p
}

func (s *Primary_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Primary_exprContext) Relation_expr() IRelation_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IRelation_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IRelation_exprContext)
}

func (s *Primary_exprContext) LP() antlr.TerminalNode {
	return s.GetToken(SQLParserLP, 0)
}

func (s *Primary_exprContext) Logic_expr() ILogic_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(ILogic_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(ILogic_exprContext)
}

func (s *Primary_exprContext) RP() antlr.TerminalNode {
	return s.GetToken(SQLParserRP, 0)
}

func (s *Primary_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Primary_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Primary_expr() (localctx IPrimary_exprContext) {
	localctx = NewPrimary_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 8, SQLParserRULE_primary_expr)
	p.SetState(64)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetTokenStream().LA(1) {
	case SQLParserOR, SQLParserAND, SQLParserNOT, SQLParserIN, SQLParserBETWEEN, SQLParserLIKE, SQLParserWHERE, SQLParserSELECT, SQLParserAS, SQLParserBY, SQLParserORDER, SQLParserASC, SQLParserDESC, SQLParserLIMIT, SQLParserREGULAR_ID:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(59)
			p.Relation_expr()
		}

	case SQLParserLP:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(60)
			p.Match(SQLParserLP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(61)
			p.Logic_expr()
		}
		{
			p.SetState(62)
			p.Match(SQLParserRP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	default:
		p.SetError(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IRelation_exprContext is an interface to support dynamic dispatch.
type IRelation_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Identifier() IIdentifierContext
	Rel_oper() IRel_operContext
	Value_expr() IValue_exprContext
	LIKE() antlr.TerminalNode
	IN() antlr.TerminalNode
	LP() antlr.TerminalNode
	In_value_expr_list() IIn_value_expr_listContext
	RP() antlr.TerminalNode
	NOT() antlr.TerminalNode
	CONTAIN_ALL() antlr.TerminalNode
	CONTAIN_ANY() antlr.TerminalNode
	IS() antlr.TerminalNode
	NULL_V() antlr.TerminalNode
	Function_call() IFunction_callContext

	// IsRelation_exprContext differentiates from other interfaces.
	IsRelation_exprContext()
}

type Relation_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyRelation_exprContext() *Relation_exprContext {
	var p = new(Relation_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_relation_expr
	return p
}

func InitEmptyRelation_exprContext(p *Relation_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_relation_expr
}

func (*Relation_exprContext) IsRelation_exprContext() {}

func NewRelation_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Relation_exprContext {
	var p = new(Relation_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_relation_expr

	return p
}

func (s *Relation_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Relation_exprContext) Identifier() IIdentifierContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IIdentifierContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IIdentifierContext)
}

func (s *Relation_exprContext) Rel_oper() IRel_operContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IRel_operContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IRel_operContext)
}

func (s *Relation_exprContext) Value_expr() IValue_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IValue_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IValue_exprContext)
}

func (s *Relation_exprContext) LIKE() antlr.TerminalNode {
	return s.GetToken(SQLParserLIKE, 0)
}

func (s *Relation_exprContext) IN() antlr.TerminalNode {
	return s.GetToken(SQLParserIN, 0)
}

func (s *Relation_exprContext) LP() antlr.TerminalNode {
	return s.GetToken(SQLParserLP, 0)
}

func (s *Relation_exprContext) In_value_expr_list() IIn_value_expr_listContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IIn_value_expr_listContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IIn_value_expr_listContext)
}

func (s *Relation_exprContext) RP() antlr.TerminalNode {
	return s.GetToken(SQLParserRP, 0)
}

func (s *Relation_exprContext) NOT() antlr.TerminalNode {
	return s.GetToken(SQLParserNOT, 0)
}

func (s *Relation_exprContext) CONTAIN_ALL() antlr.TerminalNode {
	return s.GetToken(SQLParserCONTAIN_ALL, 0)
}

func (s *Relation_exprContext) CONTAIN_ANY() antlr.TerminalNode {
	return s.GetToken(SQLParserCONTAIN_ANY, 0)
}

func (s *Relation_exprContext) IS() antlr.TerminalNode {
	return s.GetToken(SQLParserIS, 0)
}

func (s *Relation_exprContext) NULL_V() antlr.TerminalNode {
	return s.GetToken(SQLParserNULL_V, 0)
}

func (s *Relation_exprContext) Function_call() IFunction_callContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IFunction_callContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IFunction_callContext)
}

func (s *Relation_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Relation_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Relation_expr() (localctx IRelation_exprContext) {
	localctx = NewRelation_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 10, SQLParserRULE_relation_expr)
	var _la int

	p.SetState(105)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 7, p.GetParserRuleContext()) {
	case 1:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(66)
			p.Identifier()
		}
		{
			p.SetState(67)
			p.Rel_oper()
		}
		{
			p.SetState(68)
			p.Value_expr()
		}

	case 2:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(70)
			p.Identifier()
		}
		{
			p.SetState(71)
			p.Match(SQLParserLIKE)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(72)
			p.Value_expr()
		}

	case 3:
		p.EnterOuterAlt(localctx, 3)
		{
			p.SetState(74)
			p.Identifier()
		}
		p.SetState(76)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)

		if _la == SQLParserNOT {
			{
				p.SetState(75)
				p.Match(SQLParserNOT)
				if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
				}
			}

		}
		{
			p.SetState(78)
			p.Match(SQLParserIN)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(79)
			p.Match(SQLParserLP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(80)
			p.In_value_expr_list()
		}
		{
			p.SetState(81)
			p.Match(SQLParserRP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 4:
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(83)
			p.Identifier()
		}
		p.SetState(85)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)

		if _la == SQLParserNOT {
			{
				p.SetState(84)
				p.Match(SQLParserNOT)
				if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
				}
			}

		}
		{
			p.SetState(87)
			_la = p.GetTokenStream().LA(1)

			if !(_la == SQLParserCONTAIN_ALL || _la == SQLParserCONTAIN_ANY) {
				p.GetErrorHandler().RecoverInline(p)
			} else {
				p.GetErrorHandler().ReportMatch(p)
				p.Consume()
			}
		}
		{
			p.SetState(88)
			p.Match(SQLParserLP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		p.SetState(90)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)

		if (int64(_la) & ^0x3f) == 0 && ((int64(1)<<_la)&63700992) != 0 {
			{
				p.SetState(89)
				p.In_value_expr_list()
			}

		}
		{
			p.SetState(92)
			p.Match(SQLParserRP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 5:
		p.EnterOuterAlt(localctx, 5)
		{
			p.SetState(94)
			p.Identifier()
		}
		{
			p.SetState(95)
			p.Match(SQLParserIS)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		p.SetState(97)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)

		if _la == SQLParserNOT {
			{
				p.SetState(96)
				p.Match(SQLParserNOT)
				if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
				}
			}

		}
		{
			p.SetState(99)
			p.Match(SQLParserNULL_V)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 6:
		p.EnterOuterAlt(localctx, 6)
		{
			p.SetState(101)
			p.Function_call()
		}
		{
			p.SetState(102)
			p.Rel_oper()
		}
		{
			p.SetState(103)
			p.Value_expr()
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IRel_operContext is an interface to support dynamic dispatch.
type IRel_operContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	E_OP() antlr.TerminalNode
	NE_OP() antlr.TerminalNode
	L_OP() antlr.TerminalNode
	G_OP() antlr.TerminalNode
	LE_OP() antlr.TerminalNode
	GE_OP() antlr.TerminalNode

	// IsRel_operContext differentiates from other interfaces.
	IsRel_operContext()
}

type Rel_operContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyRel_operContext() *Rel_operContext {
	var p = new(Rel_operContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_rel_oper
	return p
}

func InitEmptyRel_operContext(p *Rel_operContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_rel_oper
}

func (*Rel_operContext) IsRel_operContext() {}

func NewRel_operContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Rel_operContext {
	var p = new(Rel_operContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_rel_oper

	return p
}

func (s *Rel_operContext) GetParser() antlr.Parser { return s.parser }

func (s *Rel_operContext) E_OP() antlr.TerminalNode {
	return s.GetToken(SQLParserE_OP, 0)
}

func (s *Rel_operContext) NE_OP() antlr.TerminalNode {
	return s.GetToken(SQLParserNE_OP, 0)
}

func (s *Rel_operContext) L_OP() antlr.TerminalNode {
	return s.GetToken(SQLParserL_OP, 0)
}

func (s *Rel_operContext) G_OP() antlr.TerminalNode {
	return s.GetToken(SQLParserG_OP, 0)
}

func (s *Rel_operContext) LE_OP() antlr.TerminalNode {
	return s.GetToken(SQLParserLE_OP, 0)
}

func (s *Rel_operContext) GE_OP() antlr.TerminalNode {
	return s.GetToken(SQLParserGE_OP, 0)
}

func (s *Rel_operContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Rel_operContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Rel_oper() (localctx IRel_operContext) {
	localctx = NewRel_operContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 12, SQLParserRULE_rel_oper)
	p.SetState(117)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 8, p.GetParserRuleContext()) {
	case 1:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(107)
			p.Match(SQLParserE_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 2:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(108)
			p.Match(SQLParserNE_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 3:
		p.EnterOuterAlt(localctx, 3)
		{
			p.SetState(109)
			p.Match(SQLParserL_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 4:
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(110)
			p.Match(SQLParserG_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 5:
		p.EnterOuterAlt(localctx, 5)
		{
			p.SetState(111)
			p.Match(SQLParserL_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(112)
			p.Match(SQLParserE_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 6:
		p.EnterOuterAlt(localctx, 6)
		{
			p.SetState(113)
			p.Match(SQLParserG_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(114)
			p.Match(SQLParserE_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 7:
		p.EnterOuterAlt(localctx, 7)
		{
			p.SetState(115)
			p.Match(SQLParserLE_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case 8:
		p.EnterOuterAlt(localctx, 8)
		{
			p.SetState(116)
			p.Match(SQLParserGE_OP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IValue_exprContext is an interface to support dynamic dispatch.
type IValue_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Constant() IConstantContext
	Function_call() IFunction_callContext

	// IsValue_exprContext differentiates from other interfaces.
	IsValue_exprContext()
}

type Value_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyValue_exprContext() *Value_exprContext {
	var p = new(Value_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_value_expr
	return p
}

func InitEmptyValue_exprContext(p *Value_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_value_expr
}

func (*Value_exprContext) IsValue_exprContext() {}

func NewValue_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Value_exprContext {
	var p = new(Value_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_value_expr

	return p
}

func (s *Value_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Value_exprContext) Constant() IConstantContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IConstantContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IConstantContext)
}

func (s *Value_exprContext) Function_call() IFunction_callContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IFunction_callContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IFunction_callContext)
}

func (s *Value_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Value_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Value_expr() (localctx IValue_exprContext) {
	localctx = NewValue_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 14, SQLParserRULE_value_expr)
	p.SetState(121)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetTokenStream().LA(1) {
	case SQLParserTRUE_V, SQLParserFALSE_V, SQLParserINTEGER, SQLParserFLOAT, SQLParserSQUOTA_STRING, SQLParserDQUOTA_STRING, SQLParserLMP:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(119)
			p.Constant()
		}

	case SQLParserOR, SQLParserAND, SQLParserNOT, SQLParserIN, SQLParserBETWEEN, SQLParserLIKE, SQLParserWHERE, SQLParserSELECT, SQLParserAS, SQLParserBY, SQLParserORDER, SQLParserASC, SQLParserDESC, SQLParserLIMIT, SQLParserREGULAR_ID:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(120)
			p.Function_call()
		}

	default:
		p.SetError(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IIn_value_expr_listContext is an interface to support dynamic dispatch.
type IIn_value_expr_listContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	AllIn_value_expr() []IIn_value_exprContext
	In_value_expr(i int) IIn_value_exprContext
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsIn_value_expr_listContext differentiates from other interfaces.
	IsIn_value_expr_listContext()
}

type In_value_expr_listContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyIn_value_expr_listContext() *In_value_expr_listContext {
	var p = new(In_value_expr_listContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_in_value_expr_list
	return p
}

func InitEmptyIn_value_expr_listContext(p *In_value_expr_listContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_in_value_expr_list
}

func (*In_value_expr_listContext) IsIn_value_expr_listContext() {}

func NewIn_value_expr_listContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *In_value_expr_listContext {
	var p = new(In_value_expr_listContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_in_value_expr_list

	return p
}

func (s *In_value_expr_listContext) GetParser() antlr.Parser { return s.parser }

func (s *In_value_expr_listContext) AllIn_value_expr() []IIn_value_exprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IIn_value_exprContext); ok {
			len++
		}
	}

	tst := make([]IIn_value_exprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IIn_value_exprContext); ok {
			tst[i] = t.(IIn_value_exprContext)
			i++
		}
	}

	return tst
}

func (s *In_value_expr_listContext) In_value_expr(i int) IIn_value_exprContext {
	var t antlr.RuleContext
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IIn_value_exprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext)
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IIn_value_exprContext)
}

func (s *In_value_expr_listContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(SQLParserCOMMA)
}

func (s *In_value_expr_listContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(SQLParserCOMMA, i)
}

func (s *In_value_expr_listContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *In_value_expr_listContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) In_value_expr_list() (localctx IIn_value_expr_listContext) {
	localctx = NewIn_value_expr_listContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 16, SQLParserRULE_in_value_expr_list)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(123)
		p.In_value_expr()
	}
	p.SetState(128)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)

	for _la == SQLParserCOMMA {
		{
			p.SetState(124)
			p.Match(SQLParserCOMMA)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(125)
			p.In_value_expr()
		}

		p.SetState(130)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IIn_value_exprContext is an interface to support dynamic dispatch.
type IIn_value_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Numeric() INumericContext
	Quoted_string() IQuoted_stringContext
	Bool_value() IBool_valueContext

	// IsIn_value_exprContext differentiates from other interfaces.
	IsIn_value_exprContext()
}

type In_value_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyIn_value_exprContext() *In_value_exprContext {
	var p = new(In_value_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_in_value_expr
	return p
}

func InitEmptyIn_value_exprContext(p *In_value_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_in_value_expr
}

func (*In_value_exprContext) IsIn_value_exprContext() {}

func NewIn_value_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *In_value_exprContext {
	var p = new(In_value_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_in_value_expr

	return p
}

func (s *In_value_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *In_value_exprContext) Numeric() INumericContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(INumericContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(INumericContext)
}

func (s *In_value_exprContext) Quoted_string() IQuoted_stringContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IQuoted_stringContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IQuoted_stringContext)
}

func (s *In_value_exprContext) Bool_value() IBool_valueContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IBool_valueContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IBool_valueContext)
}

func (s *In_value_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *In_value_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) In_value_expr() (localctx IIn_value_exprContext) {
	localctx = NewIn_value_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 18, SQLParserRULE_in_value_expr)
	p.SetState(134)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetTokenStream().LA(1) {
	case SQLParserINTEGER, SQLParserFLOAT:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(131)
			p.Numeric()
		}

	case SQLParserSQUOTA_STRING, SQLParserDQUOTA_STRING:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(132)
			p.Quoted_string()
		}

	case SQLParserTRUE_V, SQLParserFALSE_V:
		p.EnterOuterAlt(localctx, 3)
		{
			p.SetState(133)
			p.Bool_value()
		}

	default:
		p.SetError(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IConstantContext is an interface to support dynamic dispatch.
type IConstantContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Numeric() INumericContext
	Quoted_string() IQuoted_stringContext
	Vector_expr() IVector_exprContext
	Bool_value() IBool_valueContext

	// IsConstantContext differentiates from other interfaces.
	IsConstantContext()
}

type ConstantContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyConstantContext() *ConstantContext {
	var p = new(ConstantContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_constant
	return p
}

func InitEmptyConstantContext(p *ConstantContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_constant
}

func (*ConstantContext) IsConstantContext() {}

func NewConstantContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *ConstantContext {
	var p = new(ConstantContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_constant

	return p
}

func (s *ConstantContext) GetParser() antlr.Parser { return s.parser }

func (s *ConstantContext) Numeric() INumericContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(INumericContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(INumericContext)
}

func (s *ConstantContext) Quoted_string() IQuoted_stringContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IQuoted_stringContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IQuoted_stringContext)
}

func (s *ConstantContext) Vector_expr() IVector_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IVector_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IVector_exprContext)
}

func (s *ConstantContext) Bool_value() IBool_valueContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IBool_valueContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IBool_valueContext)
}

func (s *ConstantContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *ConstantContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Constant() (localctx IConstantContext) {
	localctx = NewConstantContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 20, SQLParserRULE_constant)
	p.SetState(140)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetTokenStream().LA(1) {
	case SQLParserINTEGER, SQLParserFLOAT:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(136)
			p.Numeric()
		}

	case SQLParserSQUOTA_STRING, SQLParserDQUOTA_STRING:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(137)
			p.Quoted_string()
		}

	case SQLParserLMP:
		p.EnterOuterAlt(localctx, 3)
		{
			p.SetState(138)
			p.Vector_expr()
		}

	case SQLParserTRUE_V, SQLParserFALSE_V:
		p.EnterOuterAlt(localctx, 4)
		{
			p.SetState(139)
			p.Bool_value()
		}

	default:
		p.SetError(antlr.NewNoViableAltException(p, nil, nil, nil, nil, nil))
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IVector_exprContext is an interface to support dynamic dispatch.
type IVector_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	AllVector() []IVectorContext
	Vector(i int) IVectorContext
	LMP() antlr.TerminalNode
	RMP() antlr.TerminalNode
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsVector_exprContext differentiates from other interfaces.
	IsVector_exprContext()
}

type Vector_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyVector_exprContext() *Vector_exprContext {
	var p = new(Vector_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_vector_expr
	return p
}

func InitEmptyVector_exprContext(p *Vector_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_vector_expr
}

func (*Vector_exprContext) IsVector_exprContext() {}

func NewVector_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Vector_exprContext {
	var p = new(Vector_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_vector_expr

	return p
}

func (s *Vector_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Vector_exprContext) AllVector() []IVectorContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IVectorContext); ok {
			len++
		}
	}

	tst := make([]IVectorContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IVectorContext); ok {
			tst[i] = t.(IVectorContext)
			i++
		}
	}

	return tst
}

func (s *Vector_exprContext) Vector(i int) IVectorContext {
	var t antlr.RuleContext
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IVectorContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext)
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IVectorContext)
}

func (s *Vector_exprContext) LMP() antlr.TerminalNode {
	return s.GetToken(SQLParserLMP, 0)
}

func (s *Vector_exprContext) RMP() antlr.TerminalNode {
	return s.GetToken(SQLParserRMP, 0)
}

func (s *Vector_exprContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(SQLParserCOMMA)
}

func (s *Vector_exprContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(SQLParserCOMMA, i)
}

func (s *Vector_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Vector_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Vector_expr() (localctx IVector_exprContext) {
	localctx = NewVector_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 22, SQLParserRULE_vector_expr)
	var _la int

	p.SetState(154)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 14, p.GetParserRuleContext()) {
	case 1:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(142)
			p.Vector()
		}

	case 2:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(143)
			p.Match(SQLParserLMP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(144)
			p.Vector()
		}
		p.SetState(149)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)

		for _la == SQLParserCOMMA {
			{
				p.SetState(145)
				p.Match(SQLParserCOMMA)
				if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
				}
			}
			{
				p.SetState(146)
				p.Vector()
			}

			p.SetState(151)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
				goto errorExit
			}
			_la = p.GetTokenStream().LA(1)
		}
		{
			p.SetState(152)
			p.Match(SQLParserRMP)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IVectorContext is an interface to support dynamic dispatch.
type IVectorContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	LMP() antlr.TerminalNode
	AllNumeric() []INumericContext
	Numeric(i int) INumericContext
	RMP() antlr.TerminalNode
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsVectorContext differentiates from other interfaces.
	IsVectorContext()
}

type VectorContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyVectorContext() *VectorContext {
	var p = new(VectorContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_vector
	return p
}

func InitEmptyVectorContext(p *VectorContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_vector
}

func (*VectorContext) IsVectorContext() {}

func NewVectorContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *VectorContext {
	var p = new(VectorContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_vector

	return p
}

func (s *VectorContext) GetParser() antlr.Parser { return s.parser }

func (s *VectorContext) LMP() antlr.TerminalNode {
	return s.GetToken(SQLParserLMP, 0)
}

func (s *VectorContext) AllNumeric() []INumericContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(INumericContext); ok {
			len++
		}
	}

	tst := make([]INumericContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(INumericContext); ok {
			tst[i] = t.(INumericContext)
			i++
		}
	}

	return tst
}

func (s *VectorContext) Numeric(i int) INumericContext {
	var t antlr.RuleContext
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(INumericContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext)
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(INumericContext)
}

func (s *VectorContext) RMP() antlr.TerminalNode {
	return s.GetToken(SQLParserRMP, 0)
}

func (s *VectorContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(SQLParserCOMMA)
}

func (s *VectorContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(SQLParserCOMMA, i)
}

func (s *VectorContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *VectorContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Vector() (localctx IVectorContext) {
	localctx = NewVectorContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 24, SQLParserRULE_vector)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(156)
		p.Match(SQLParserLMP)
		if p.HasError() {
			// Recognition error - abort rule
			goto errorExit
		}
	}
	{
		p.SetState(157)
		p.Numeric()
	}
	p.SetState(162)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)

	for _la == SQLParserCOMMA {
		{
			p.SetState(158)
			p.Match(SQLParserCOMMA)
			if p.HasError() {
				// Recognition error - abort rule
				goto errorExit
			}
		}
		{
			p.SetState(159)
			p.Numeric()
		}

		p.SetState(164)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)
	}
	{
		p.SetState(165)
		p.Match(SQLParserRMP)
		if p.HasError() {
			// Recognition error - abort rule
			goto errorExit
		}
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IFunction_value_exprContext is an interface to support dynamic dispatch.
type IFunction_value_exprContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Value_expr() IValue_exprContext
	Identifier() IIdentifierContext

	// IsFunction_value_exprContext differentiates from other interfaces.
	IsFunction_value_exprContext()
}

type Function_value_exprContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyFunction_value_exprContext() *Function_value_exprContext {
	var p = new(Function_value_exprContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_function_value_expr
	return p
}

func InitEmptyFunction_value_exprContext(p *Function_value_exprContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_function_value_expr
}

func (*Function_value_exprContext) IsFunction_value_exprContext() {}

func NewFunction_value_exprContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Function_value_exprContext {
	var p = new(Function_value_exprContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_function_value_expr

	return p
}

func (s *Function_value_exprContext) GetParser() antlr.Parser { return s.parser }

func (s *Function_value_exprContext) Value_expr() IValue_exprContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IValue_exprContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IValue_exprContext)
}

func (s *Function_value_exprContext) Identifier() IIdentifierContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IIdentifierContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IIdentifierContext)
}

func (s *Function_value_exprContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Function_value_exprContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Function_value_expr() (localctx IFunction_value_exprContext) {
	localctx = NewFunction_value_exprContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 26, SQLParserRULE_function_value_expr)
	p.SetState(169)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}

	switch p.GetInterpreter().AdaptivePredict(p.BaseParser, p.GetTokenStream(), 16, p.GetParserRuleContext()) {
	case 1:
		p.EnterOuterAlt(localctx, 1)
		{
			p.SetState(167)
			p.Value_expr()
		}

	case 2:
		p.EnterOuterAlt(localctx, 2)
		{
			p.SetState(168)
			p.Identifier()
		}

	case antlr.ATNInvalidAltNumber:
		goto errorExit
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IFunction_callContext is an interface to support dynamic dispatch.
type IFunction_callContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	Identifier() IIdentifierContext
	LP() antlr.TerminalNode
	RP() antlr.TerminalNode
	AllFunction_value_expr() []IFunction_value_exprContext
	Function_value_expr(i int) IFunction_value_exprContext
	AllCOMMA() []antlr.TerminalNode
	COMMA(i int) antlr.TerminalNode

	// IsFunction_callContext differentiates from other interfaces.
	IsFunction_callContext()
}

type Function_callContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyFunction_callContext() *Function_callContext {
	var p = new(Function_callContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_function_call
	return p
}

func InitEmptyFunction_callContext(p *Function_callContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_function_call
}

func (*Function_callContext) IsFunction_callContext() {}

func NewFunction_callContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Function_callContext {
	var p = new(Function_callContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_function_call

	return p
}

func (s *Function_callContext) GetParser() antlr.Parser { return s.parser }

func (s *Function_callContext) Identifier() IIdentifierContext {
	var t antlr.RuleContext
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IIdentifierContext); ok {
			t = ctx.(antlr.RuleContext)
			break
		}
	}

	if t == nil {
		return nil
	}

	return t.(IIdentifierContext)
}

func (s *Function_callContext) LP() antlr.TerminalNode {
	return s.GetToken(SQLParserLP, 0)
}

func (s *Function_callContext) RP() antlr.TerminalNode {
	return s.GetToken(SQLParserRP, 0)
}

func (s *Function_callContext) AllFunction_value_expr() []IFunction_value_exprContext {
	children := s.GetChildren()
	len := 0
	for _, ctx := range children {
		if _, ok := ctx.(IFunction_value_exprContext); ok {
			len++
		}
	}

	tst := make([]IFunction_value_exprContext, len)
	i := 0
	for _, ctx := range children {
		if t, ok := ctx.(IFunction_value_exprContext); ok {
			tst[i] = t.(IFunction_value_exprContext)
			i++
		}
	}

	return tst
}

func (s *Function_callContext) Function_value_expr(i int) IFunction_value_exprContext {
	var t antlr.RuleContext
	j := 0
	for _, ctx := range s.GetChildren() {
		if _, ok := ctx.(IFunction_value_exprContext); ok {
			if j == i {
				t = ctx.(antlr.RuleContext)
				break
			}
			j++
		}
	}

	if t == nil {
		return nil
	}

	return t.(IFunction_value_exprContext)
}

func (s *Function_callContext) AllCOMMA() []antlr.TerminalNode {
	return s.GetTokens(SQLParserCOMMA)
}

func (s *Function_callContext) COMMA(i int) antlr.TerminalNode {
	return s.GetToken(SQLParserCOMMA, i)
}

func (s *Function_callContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Function_callContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Function_call() (localctx IFunction_callContext) {
	localctx = NewFunction_callContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 28, SQLParserRULE_function_call)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(171)
		p.Identifier()
	}
	{
		p.SetState(172)
		p.Match(SQLParserLP)
		if p.HasError() {
			// Recognition error - abort rule
			goto errorExit
		}
	}
	p.SetState(181)
	p.GetErrorHandler().Sync(p)
	if p.HasError() {
		goto errorExit
	}
	_la = p.GetTokenStream().LA(1)

	if (int64(_la) & ^0x3f) == 0 && ((int64(1)<<_la)&4398378907550) != 0 {
		{
			p.SetState(173)
			p.Function_value_expr()
		}
		p.SetState(178)
		p.GetErrorHandler().Sync(p)
		if p.HasError() {
			goto errorExit
		}
		_la = p.GetTokenStream().LA(1)

		for _la == SQLParserCOMMA {
			{
				p.SetState(174)
				p.Match(SQLParserCOMMA)
				if p.HasError() {
					// Recognition error - abort rule
					goto errorExit
				}
			}
			{
				p.SetState(175)
				p.Function_value_expr()
			}

			p.SetState(180)
			p.GetErrorHandler().Sync(p)
			if p.HasError() {
				goto errorExit
			}
			_la = p.GetTokenStream().LA(1)
		}

	}
	{
		p.SetState(183)
		p.Match(SQLParserRP)
		if p.HasError() {
			// Recognition error - abort rule
			goto errorExit
		}
	}

errorExit:
	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// INumericContext is an interface to support dynamic dispatch.
type INumericContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	INTEGER() antlr.TerminalNode
	FLOAT() antlr.TerminalNode

	// IsNumericContext differentiates from other interfaces.
	IsNumericContext()
}

type NumericContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyNumericContext() *NumericContext {
	var p = new(NumericContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_numeric
	return p
}

func InitEmptyNumericContext(p *NumericContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_numeric
}

func (*NumericContext) IsNumericContext() {}

func NewNumericContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *NumericContext {
	var p = new(NumericContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_numeric

	return p
}

func (s *NumericContext) GetParser() antlr.Parser { return s.parser }

func (s *NumericContext) INTEGER() antlr.TerminalNode {
	return s.GetToken(SQLParserINTEGER, 0)
}

func (s *NumericContext) FLOAT() antlr.TerminalNode {
	return s.GetToken(SQLParserFLOAT, 0)
}

func (s *NumericContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *NumericContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Numeric() (localctx INumericContext) {
	localctx = NewNumericContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 30, SQLParserRULE_numeric)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(185)
		_la = p.GetTokenStream().LA(1)

		if !(_la == SQLParserINTEGER || _la == SQLParserFLOAT) {
			p.GetErrorHandler().RecoverInline(p)
		} else {
			p.GetErrorHandler().ReportMatch(p)
			p.Consume()
		}
	}

	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IQuoted_stringContext is an interface to support dynamic dispatch.
type IQuoted_stringContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	SQUOTA_STRING() antlr.TerminalNode
	DQUOTA_STRING() antlr.TerminalNode

	// IsQuoted_stringContext differentiates from other interfaces.
	IsQuoted_stringContext()
}

type Quoted_stringContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyQuoted_stringContext() *Quoted_stringContext {
	var p = new(Quoted_stringContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_quoted_string
	return p
}

func InitEmptyQuoted_stringContext(p *Quoted_stringContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_quoted_string
}

func (*Quoted_stringContext) IsQuoted_stringContext() {}

func NewQuoted_stringContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Quoted_stringContext {
	var p = new(Quoted_stringContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_quoted_string

	return p
}

func (s *Quoted_stringContext) GetParser() antlr.Parser { return s.parser }

func (s *Quoted_stringContext) SQUOTA_STRING() antlr.TerminalNode {
	return s.GetToken(SQLParserSQUOTA_STRING, 0)
}

func (s *Quoted_stringContext) DQUOTA_STRING() antlr.TerminalNode {
	return s.GetToken(SQLParserDQUOTA_STRING, 0)
}

func (s *Quoted_stringContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Quoted_stringContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Quoted_string() (localctx IQuoted_stringContext) {
	localctx = NewQuoted_stringContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 32, SQLParserRULE_quoted_string)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(187)
		_la = p.GetTokenStream().LA(1)

		if !(_la == SQLParserSQUOTA_STRING || _la == SQLParserDQUOTA_STRING) {
			p.GetErrorHandler().RecoverInline(p)
		} else {
			p.GetErrorHandler().ReportMatch(p)
			p.Consume()
		}
	}

	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IBool_valueContext is an interface to support dynamic dispatch.
type IBool_valueContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	TRUE_V() antlr.TerminalNode
	FALSE_V() antlr.TerminalNode

	// IsBool_valueContext differentiates from other interfaces.
	IsBool_valueContext()
}

type Bool_valueContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyBool_valueContext() *Bool_valueContext {
	var p = new(Bool_valueContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_bool_value
	return p
}

func InitEmptyBool_valueContext(p *Bool_valueContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_bool_value
}

func (*Bool_valueContext) IsBool_valueContext() {}

func NewBool_valueContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *Bool_valueContext {
	var p = new(Bool_valueContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_bool_value

	return p
}

func (s *Bool_valueContext) GetParser() antlr.Parser { return s.parser }

func (s *Bool_valueContext) TRUE_V() antlr.TerminalNode {
	return s.GetToken(SQLParserTRUE_V, 0)
}

func (s *Bool_valueContext) FALSE_V() antlr.TerminalNode {
	return s.GetToken(SQLParserFALSE_V, 0)
}

func (s *Bool_valueContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *Bool_valueContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Bool_value() (localctx IBool_valueContext) {
	localctx = NewBool_valueContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 34, SQLParserRULE_bool_value)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(189)
		_la = p.GetTokenStream().LA(1)

		if !(_la == SQLParserTRUE_V || _la == SQLParserFALSE_V) {
			p.GetErrorHandler().RecoverInline(p)
		} else {
			p.GetErrorHandler().ReportMatch(p)
			p.Consume()
		}
	}

	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}

// IIdentifierContext is an interface to support dynamic dispatch.
type IIdentifierContext interface {
	antlr.ParserRuleContext

	// GetParser returns the parser.
	GetParser() antlr.Parser

	// Getter signatures
	REGULAR_ID() antlr.TerminalNode
	OR() antlr.TerminalNode
	AND() antlr.TerminalNode
	NOT() antlr.TerminalNode
	IN() antlr.TerminalNode
	BETWEEN() antlr.TerminalNode
	LIKE() antlr.TerminalNode
	WHERE() antlr.TerminalNode
	SELECT() antlr.TerminalNode
	AS() antlr.TerminalNode
	BY() antlr.TerminalNode
	ORDER() antlr.TerminalNode
	ASC() antlr.TerminalNode
	DESC() antlr.TerminalNode
	LIMIT() antlr.TerminalNode

	// IsIdentifierContext differentiates from other interfaces.
	IsIdentifierContext()
}

type IdentifierContext struct {
	antlr.BaseParserRuleContext
	parser antlr.Parser
}

func NewEmptyIdentifierContext() *IdentifierContext {
	var p = new(IdentifierContext)
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_identifier
	return p
}

func InitEmptyIdentifierContext(p *IdentifierContext) {
	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, nil, -1)
	p.RuleIndex = SQLParserRULE_identifier
}

func (*IdentifierContext) IsIdentifierContext() {}

func NewIdentifierContext(parser antlr.Parser, parent antlr.ParserRuleContext, invokingState int) *IdentifierContext {
	var p = new(IdentifierContext)

	antlr.InitBaseParserRuleContext(&p.BaseParserRuleContext, parent, invokingState)

	p.parser = parser
	p.RuleIndex = SQLParserRULE_identifier

	return p
}

func (s *IdentifierContext) GetParser() antlr.Parser { return s.parser }

func (s *IdentifierContext) REGULAR_ID() antlr.TerminalNode {
	return s.GetToken(SQLParserREGULAR_ID, 0)
}

func (s *IdentifierContext) OR() antlr.TerminalNode {
	return s.GetToken(SQLParserOR, 0)
}

func (s *IdentifierContext) AND() antlr.TerminalNode {
	return s.GetToken(SQLParserAND, 0)
}

func (s *IdentifierContext) NOT() antlr.TerminalNode {
	return s.GetToken(SQLParserNOT, 0)
}

func (s *IdentifierContext) IN() antlr.TerminalNode {
	return s.GetToken(SQLParserIN, 0)
}

func (s *IdentifierContext) BETWEEN() antlr.TerminalNode {
	return s.GetToken(SQLParserBETWEEN, 0)
}

func (s *IdentifierContext) LIKE() antlr.TerminalNode {
	return s.GetToken(SQLParserLIKE, 0)
}

func (s *IdentifierContext) WHERE() antlr.TerminalNode {
	return s.GetToken(SQLParserWHERE, 0)
}

func (s *IdentifierContext) SELECT() antlr.TerminalNode {
	return s.GetToken(SQLParserSELECT, 0)
}

func (s *IdentifierContext) AS() antlr.TerminalNode {
	return s.GetToken(SQLParserAS, 0)
}

func (s *IdentifierContext) BY() antlr.TerminalNode {
	return s.GetToken(SQLParserBY, 0)
}

func (s *IdentifierContext) ORDER() antlr.TerminalNode {
	return s.GetToken(SQLParserORDER, 0)
}

func (s *IdentifierContext) ASC() antlr.TerminalNode {
	return s.GetToken(SQLParserASC, 0)
}

func (s *IdentifierContext) DESC() antlr.TerminalNode {
	return s.GetToken(SQLParserDESC, 0)
}

func (s *IdentifierContext) LIMIT() antlr.TerminalNode {
	return s.GetToken(SQLParserLIMIT, 0)
}

func (s *IdentifierContext) GetRuleContext() antlr.RuleContext {
	return s
}

func (s *IdentifierContext) ToStringTree(ruleNames []string, recog antlr.Recognizer) string {
	return antlr.TreesStringTree(s, ruleNames, recog)
}

func (p *SQLParser) Identifier() (localctx IIdentifierContext) {
	localctx = NewIdentifierContext(p, p.GetParserRuleContext(), p.GetState())
	p.EnterRule(localctx, 36, SQLParserRULE_identifier)
	var _la int

	p.EnterOuterAlt(localctx, 1)
	{
		p.SetState(191)
		_la = p.GetTokenStream().LA(1)

		if !((int64(_la) & ^0x3f) == 0 && ((int64(1)<<_la)&4398046771102) != 0) {
			p.GetErrorHandler().RecoverInline(p)
		} else {
			p.GetErrorHandler().ReportMatch(p)
			p.Consume()
		}
	}

	if p.HasError() {
		v := p.GetError()
		localctx.SetException(v)
		p.GetErrorHandler().ReportError(p, v)
		p.GetErrorHandler().Recover(p, v)
		p.SetError(nil)
	}
	p.ExitRule()
	return localctx
}
