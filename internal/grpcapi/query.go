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

package grpcapi

import (
	"fmt"

	"github.com/gorse-io/xvec"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
)

func RerankerToProto(reranker xvec.Reranker) (*xvecv1.Reranker, error) {
	if reranker == nil {
		return nil, nil
	}
	switch r := reranker.(type) {
	case xvec.RRFReranker:
		return &xvecv1.Reranker{Kind: &xvecv1.Reranker_Rrf{Rrf: &xvecv1.RRFReranker{RankConstant: int64(r.RankConstant)}}}, nil
	case *xvec.RRFReranker:
		if r != nil {
			return RerankerToProto(*r)
		}
	case xvec.WeightedReranker:
		return &xvecv1.Reranker{Kind: &xvecv1.Reranker_Weighted{Weighted: &xvecv1.WeightedReranker{Weights: append([]float64(nil), r.Weights...)}}}, nil
	case *xvec.WeightedReranker:
		if r != nil {
			return RerankerToProto(*r)
		}
	}
	return nil, &xvec.Error{Code: xvec.ErrorCodeNotSupported, Op: "encode protobuf", Message: fmt.Sprintf("unsupported reranker %T", reranker)}
}

func RerankerFromProto(message *xvecv1.Reranker) (xvec.Reranker, error) {
	if message == nil {
		return nil, nil
	}
	switch r := message.Kind.(type) {
	case *xvecv1.Reranker_Rrf:
		if r.Rrf != nil {
			rankConstant, err := nonNegativeIntFromProto("RRF rank constant", r.Rrf.RankConstant)
			if err != nil {
				return nil, err
			}
			return xvec.RRFReranker{RankConstant: rankConstant}, nil
		}
	case *xvecv1.Reranker_Weighted:
		if r.Weighted != nil {
			return xvec.WeightedReranker{Weights: append([]float64(nil), r.Weighted.Weights...)}, nil
		}
	}
	return nil, invalid("reranker kind is missing")
}

func ftsClauseToProto(clause *xvec.FTSClause) (*xvecv1.FTSClause, error) {
	if clause == nil {
		return nil, nil
	}
	if clause.Query != "" && clause.Match != "" {
		return nil, invalid("FTS query and match are mutually exclusive")
	}
	if clause.Query != "" {
		return &xvecv1.FTSClause{Kind: &xvecv1.FTSClause_Query{Query: clause.Query}}, nil
	}
	return &xvecv1.FTSClause{Kind: &xvecv1.FTSClause_Match{Match: clause.Match}}, nil
}

func ftsClauseFromProto(message *xvecv1.FTSClause) (*xvec.FTSClause, error) {
	if message == nil {
		return nil, invalid("FTS clause is nil")
	}
	switch clause := message.Kind.(type) {
	case *xvecv1.FTSClause_Query:
		return &xvec.FTSClause{Query: clause.Query}, nil
	case *xvecv1.FTSClause_Match:
		return &xvec.FTSClause{Match: clause.Match}, nil
	default:
		return nil, invalid("FTS clause kind is missing")
	}
}

func queryTargetToProto(dense xvec.DenseVector, sparse xvec.SparseVector, primaryKey string, fts *xvec.FTSClause, allowFilter bool) (*xvecv1.QueryTarget, error) {
	count := 0
	if dense != nil {
		count++
	}
	if sparse != nil {
		count++
	}
	if primaryKey != "" {
		count++
	}
	if fts != nil {
		count++
	}
	if count == 0 && allowFilter {
		return &xvecv1.QueryTarget{Kind: &xvecv1.QueryTarget_FilterOnly{FilterOnly: &xvecv1.FilterOnly{}}}, nil
	}
	if count != 1 {
		return nil, invalid("exactly one query target must be set")
	}
	if dense != nil {
		value, err := valueToProto(dense)
		if err != nil {
			return nil, err
		}
		return &xvecv1.QueryTarget{Kind: &xvecv1.QueryTarget_Vector{Vector: value}}, nil
	}
	if sparse != nil {
		value, err := valueToProto(sparse)
		if err != nil {
			return nil, err
		}
		return &xvecv1.QueryTarget{Kind: &xvecv1.QueryTarget_Vector{Vector: value}}, nil
	}
	if primaryKey != "" {
		return &xvecv1.QueryTarget{Kind: &xvecv1.QueryTarget_PrimaryKey{PrimaryKey: primaryKey}}, nil
	}
	clause, err := ftsClauseToProto(fts)
	if err != nil {
		return nil, err
	}
	return &xvecv1.QueryTarget{Kind: &xvecv1.QueryTarget_Fts{Fts: clause}}, nil
}

func queryTargetFromProto(message *xvecv1.QueryTarget, allowFilter bool) (xvec.DenseVector, xvec.SparseVector, string, *xvec.FTSClause, error) {
	if message == nil {
		return nil, nil, "", nil, invalid("query target is nil")
	}
	switch target := message.Kind.(type) {
	case *xvecv1.QueryTarget_Vector:
		value, err := valueFromProto(target.Vector)
		if err != nil {
			return nil, nil, "", nil, err
		}
		if dense, ok := value.(xvec.DenseVector); ok {
			return dense, nil, "", nil, nil
		}
		if sparse, ok := value.(xvec.SparseVector); ok {
			return nil, sparse, "", nil, nil
		}
		return nil, nil, "", nil, invalid("query vector has non-vector type %T", value)
	case *xvecv1.QueryTarget_PrimaryKey:
		return nil, nil, target.PrimaryKey, nil, nil
	case *xvecv1.QueryTarget_Fts:
		clause, err := ftsClauseFromProto(target.Fts)
		return nil, nil, "", clause, err
	case *xvecv1.QueryTarget_FilterOnly:
		if !allowFilter {
			return nil, nil, "", nil, invalid("filter-only target is not allowed")
		}
		return nil, nil, "", nil, nil
	default:
		return nil, nil, "", nil, invalid("query target kind is missing")
	}
}

func VectorQueryToProto(query xvec.VectorQuery) (*xvecv1.VectorQuery, error) {
	target, err := queryTargetToProto(query.DenseVector, query.SparseVector, query.PrimaryKey, query.FTS, true)
	if err != nil {
		return nil, err
	}
	params, err := QueryParamsToProto(query.Params)
	if err != nil {
		return nil, err
	}
	return &xvecv1.VectorQuery{Field: query.Field, Target: target, TopK: int64(query.TopK), Filter: query.Filter, Projection: ProjectionToProto(query.Projection), Params: params}, nil
}

func VectorQueryFromProto(message *xvecv1.VectorQuery) (xvec.VectorQuery, error) {
	if message == nil {
		return xvec.VectorQuery{}, invalid("vector query is nil")
	}
	dense, sparse, primaryKey, fts, err := queryTargetFromProto(message.Target, true)
	if err != nil {
		return xvec.VectorQuery{}, err
	}
	projection, err := ProjectionFromProto(message.Projection)
	if err != nil {
		return xvec.VectorQuery{}, err
	}
	params, err := QueryParamsFromProto(message.Params)
	if err != nil {
		return xvec.VectorQuery{}, err
	}
	topK, err := nonNegativeIntFromProto("query TopK", message.TopK)
	if err != nil {
		return xvec.VectorQuery{}, err
	}
	return xvec.VectorQuery{Field: message.Field, DenseVector: dense, SparseVector: sparse, PrimaryKey: primaryKey, FTS: fts, TopK: topK, Filter: message.Filter, Projection: projection, Params: params}, nil
}

func QueryRequestToProto(collection string, query xvec.VectorQuery) (*xvecv1.QueryRequest, error) {
	message, err := VectorQueryToProto(query)
	if err != nil {
		return nil, err
	}
	return &xvecv1.QueryRequest{Collection: collection, Query: message}, nil
}
func QueryRequestFromProto(message *xvecv1.QueryRequest) (string, xvec.VectorQuery, error) {
	if message == nil {
		return "", xvec.VectorQuery{}, invalid("query request is nil")
	}
	query, err := VectorQueryFromProto(message.Query)
	return message.Collection, query, err
}

func subQueryToProto(query xvec.SubQuery) (*xvecv1.SubQuery, error) {
	target, err := queryTargetToProto(query.DenseVector, query.SparseVector, query.PrimaryKey, query.FTS, false)
	if err != nil {
		return nil, err
	}
	params, err := QueryParamsToProto(query.Params)
	if err != nil {
		return nil, err
	}
	return &xvecv1.SubQuery{Field: query.Field, Target: target, Params: params, NumCandidates: int64(query.NumCandidates)}, nil
}
func subQueryFromProto(message *xvecv1.SubQuery) (xvec.SubQuery, error) {
	if message == nil {
		return xvec.SubQuery{}, invalid("sub-query is nil")
	}
	dense, sparse, primaryKey, fts, err := queryTargetFromProto(message.Target, false)
	if err != nil {
		return xvec.SubQuery{}, err
	}
	params, err := QueryParamsFromProto(message.Params)
	if err != nil {
		return xvec.SubQuery{}, err
	}
	numCandidates, err := nonNegativeIntFromProto("sub-query NumCandidates", message.NumCandidates)
	if err != nil {
		return xvec.SubQuery{}, err
	}
	return xvec.SubQuery{Field: message.Field, DenseVector: dense, SparseVector: sparse, PrimaryKey: primaryKey, FTS: fts, Params: params, NumCandidates: numCandidates}, nil
}

func MultiQueryRequestToProto(collection string, query xvec.MultiQuery) (*xvecv1.MultiQueryRequest, error) {
	message := &xvecv1.MultiQuery{TopK: int64(query.TopK), Filter: query.Filter, Projection: ProjectionToProto(query.Projection), Queries: make([]*xvecv1.SubQuery, len(query.Queries))}
	var err error
	for i := range query.Queries {
		message.Queries[i], err = subQueryToProto(query.Queries[i])
		if err != nil {
			return nil, err
		}
	}
	message.Reranker, err = RerankerToProto(query.Reranker)
	if err != nil {
		return nil, err
	}
	return &xvecv1.MultiQueryRequest{Collection: collection, Query: message}, nil
}
func MultiQueryRequestFromProto(request *xvecv1.MultiQueryRequest) (string, xvec.MultiQuery, error) {
	if request == nil || request.Query == nil {
		return "", xvec.MultiQuery{}, invalid("multi-query request is nil")
	}
	message := request.Query
	projection, err := ProjectionFromProto(message.Projection)
	if err != nil {
		return "", xvec.MultiQuery{}, err
	}
	topK, err := nonNegativeIntFromProto("multi-query TopK", message.TopK)
	if err != nil {
		return "", xvec.MultiQuery{}, err
	}
	query := xvec.MultiQuery{TopK: topK, Filter: message.Filter, Projection: projection, Queries: make([]xvec.SubQuery, len(message.Queries))}
	for i := range message.Queries {
		query.Queries[i], err = subQueryFromProto(message.Queries[i])
		if err != nil {
			return "", xvec.MultiQuery{}, err
		}
	}
	query.Reranker, err = RerankerFromProto(message.Reranker)
	return request.Collection, query, err
}

func GroupByQueryRequestToProto(collection string, query xvec.GroupByVectorQuery) (*xvecv1.GroupByQueryRequest, error) {
	target, err := queryTargetToProto(query.DenseVector, query.SparseVector, query.PrimaryKey, nil, false)
	if err != nil {
		return nil, err
	}
	params, err := QueryParamsToProto(query.Params)
	if err != nil {
		return nil, err
	}
	message := &xvecv1.GroupByQuery{Field: query.Field, Target: target, Filter: query.Filter, Projection: ProjectionToProto(query.Projection), Params: params, GroupByField: query.GroupByField, GroupCount: int64(query.GroupCount), TopKPerGroup: int64(query.TopKPerGroup)}
	return &xvecv1.GroupByQueryRequest{Collection: collection, Query: message}, nil
}
func GroupByQueryRequestFromProto(request *xvecv1.GroupByQueryRequest) (string, xvec.GroupByVectorQuery, error) {
	if request == nil || request.Query == nil {
		return "", xvec.GroupByVectorQuery{}, invalid("group-by request is nil")
	}
	message := request.Query
	dense, sparse, primaryKey, fts, err := queryTargetFromProto(message.Target, false)
	if err != nil {
		return "", xvec.GroupByVectorQuery{}, err
	}
	if fts != nil {
		return "", xvec.GroupByVectorQuery{}, invalid("group-by does not support FTS")
	}
	projection, err := ProjectionFromProto(message.Projection)
	if err != nil {
		return "", xvec.GroupByVectorQuery{}, err
	}
	params, err := QueryParamsFromProto(message.Params)
	if err != nil {
		return "", xvec.GroupByVectorQuery{}, err
	}
	groupCount, err := nonNegativeIntFromProto("group-by GroupCount", message.GroupCount)
	if err != nil {
		return "", xvec.GroupByVectorQuery{}, err
	}
	topKPerGroup, err := nonNegativeIntFromProto("group-by TopKPerGroup", message.TopKPerGroup)
	if err != nil {
		return "", xvec.GroupByVectorQuery{}, err
	}
	query := xvec.GroupByVectorQuery{Field: message.Field, DenseVector: dense, SparseVector: sparse, PrimaryKey: primaryKey, Filter: message.Filter, Projection: projection, Params: params, GroupByField: message.GroupByField, GroupCount: groupCount, TopKPerGroup: topKPerGroup}
	return request.Collection, query, nil
}
