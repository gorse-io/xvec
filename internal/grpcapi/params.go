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
	"encoding/json"
	"fmt"

	"github.com/gorse-io/xvec"
	xvecv1 "github.com/gorse-io/xvec/internal/proto/xvec/v1"
)

func invalid(format string, args ...any) error {
	return &xvec.Error{Code: xvec.ErrorCodeInvalidArgument, Op: "decode protobuf", Message: fmt.Sprintf(format, args...)}
}

func nonNegativeIntFromProto(name string, value int64) (int, error) {
	if value < 0 {
		return 0, invalid("%s cannot be negative", name)
	}
	converted := int(value)
	if int64(converted) != value {
		return 0, invalid("%s overflows int", name)
	}
	return converted, nil
}

func IndexParamsToProto(params xvec.IndexParams) (*xvecv1.IndexParams, error) {
	if params == nil {
		return nil, nil
	}
	switch p := params.(type) {
	case xvec.InvertIndexParams:
		return invertIndexToProto(p), nil
	case *xvec.InvertIndexParams:
		if p != nil {
			return invertIndexToProto(*p), nil
		}
	case xvec.FlatIndexParams:
		return flatIndexToProto(p), nil
	case *xvec.FlatIndexParams:
		if p != nil {
			return flatIndexToProto(*p), nil
		}
	case xvec.HNSWIndexParams:
		return hnswIndexToProto(p), nil
	case *xvec.HNSWIndexParams:
		if p != nil {
			return hnswIndexToProto(*p), nil
		}
	case xvec.IVFRaBitQIndexParams:
		return ivfRaBitQIndexToProto(p), nil
	case *xvec.IVFRaBitQIndexParams:
		if p != nil {
			return ivfRaBitQIndexToProto(*p), nil
		}
	case xvec.IVFIndexParams:
		return ivfIndexToProto(p), nil
	case *xvec.IVFIndexParams:
		if p != nil {
			return ivfIndexToProto(*p), nil
		}
	case xvec.DiskANNIndexParams:
		return diskANNIndexToProto(p), nil
	case *xvec.DiskANNIndexParams:
		if p != nil {
			return diskANNIndexToProto(*p), nil
		}
	case xvec.VamanaIndexParams:
		return vamanaIndexToProto(p), nil
	case *xvec.VamanaIndexParams:
		if p != nil {
			return vamanaIndexToProto(*p), nil
		}
	case xvec.FTSIndexParams:
		return ftsIndexToProto(p), nil
	case *xvec.FTSIndexParams:
		if p != nil {
			return ftsIndexToProto(*p), nil
		}
	}
	return nil, invalid("unsupported index params %T", params)
}

func quantizerToProto(p xvec.QuantizerParams) *xvecv1.QuantizerParams {
	return &xvecv1.QuantizerParams{EnableRotate: p.EnableRotate}
}
func quantizerFromProto(p *xvecv1.QuantizerParams) xvec.QuantizerParams {
	if p == nil {
		return xvec.QuantizerParams{}
	}
	return xvec.QuantizerParams{EnableRotate: p.EnableRotate}
}
func invertIndexToProto(p xvec.InvertIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Invert{Invert: &xvecv1.InvertIndexParams{EnableRangeOptimization: p.EnableRangeOptimization, EnableExtendedWildcard: p.EnableExtendedWildcard}}}
}
func flatIndexToProto(p xvec.FlatIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Flat{Flat: &xvecv1.FlatIndexParams{Metric: xvecv1.MetricType(p.Metric), Quantize: xvecv1.QuantizeType(p.Quantize), Quantizer: quantizerToProto(p.Quantizer)}}}
}
func hnswIndexToProto(p xvec.HNSWIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Hnsw{Hnsw: &xvecv1.HNSWIndexParams{Metric: xvecv1.MetricType(p.Metric), M: int64(p.M), EfConstruction: int64(p.EFConstruction), Quantize: xvecv1.QuantizeType(p.Quantize), UseContiguousMemory: p.UseContiguousMemory, Quantizer: quantizerToProto(p.Quantizer)}}}
}
func ivfRaBitQIndexToProto(p xvec.IVFRaBitQIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_IvfRabitq{IvfRabitq: &xvecv1.IVFRaBitQIndexParams{Metric: xvecv1.MetricType(p.Metric), Nlist: int64(p.NList), TotalBits: int64(p.TotalBits), SampleCount: int64(p.SampleCount)}}}
}
func ivfIndexToProto(p xvec.IVFIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Ivf{Ivf: &xvecv1.IVFIndexParams{Metric: xvecv1.MetricType(p.Metric), Nlist: int64(p.NList), Niterations: int64(p.NIterations), UseSoar: p.UseSOAR, Quantize: xvecv1.QuantizeType(p.Quantize), Quantizer: quantizerToProto(p.Quantizer)}}}
}
func diskANNIndexToProto(p xvec.DiskANNIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Diskann{Diskann: &xvecv1.DiskANNIndexParams{Metric: xvecv1.MetricType(p.Metric), MaxDegree: int64(p.MaxDegree), ListSize: int64(p.ListSize), PqChunks: int64(p.PQChunks), Quantize: xvecv1.QuantizeType(p.Quantize), Quantizer: quantizerToProto(p.Quantizer)}}}
}
func vamanaIndexToProto(p xvec.VamanaIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Vamana{Vamana: &xvecv1.VamanaIndexParams{Metric: xvecv1.MetricType(p.Metric), MaxDegree: int64(p.MaxDegree), SearchListSize: int64(p.SearchListSize), Alpha: p.Alpha, MaxOcclusionSize: int64(p.MaxOcclusionSize), SaturateGraph: p.SaturateGraph, UseContiguousMemory: p.UseContiguousMemory, UseIdMap: p.UseIDMap, Quantize: xvecv1.QuantizeType(p.Quantize), Quantizer: quantizerToProto(p.Quantizer)}}}
}
func ftsIndexToProto(p xvec.FTSIndexParams) *xvecv1.IndexParams {
	return &xvecv1.IndexParams{Kind: &xvecv1.IndexParams_Fts{Fts: &xvecv1.FTSIndexParams{Tokenizer: p.Tokenizer, Filters: append([]string(nil), p.Filters...), ExtraParams: p.ExtraParams}}}
}

func IndexParamsFromProto(message *xvecv1.IndexParams) (xvec.IndexParams, error) {
	if message == nil {
		return nil, nil
	}
	switch p := message.Kind.(type) {
	case *xvecv1.IndexParams_Invert:
		if p.Invert == nil {
			return nil, invalid("nil invert index params")
		}
		return xvec.InvertIndexParams{EnableRangeOptimization: p.Invert.EnableRangeOptimization, EnableExtendedWildcard: p.Invert.EnableExtendedWildcard}, nil
	case *xvecv1.IndexParams_Flat:
		if p.Flat == nil {
			return nil, invalid("nil flat index params")
		}
		return xvec.FlatIndexParams{Metric: xvec.MetricType(p.Flat.Metric), Quantize: xvec.QuantizeType(p.Flat.Quantize), Quantizer: quantizerFromProto(p.Flat.Quantizer)}, nil
	case *xvecv1.IndexParams_Hnsw:
		if p.Hnsw == nil {
			return nil, invalid("nil HNSW index params")
		}
		m, err := nonNegativeIntFromProto("HNSW M", p.Hnsw.M)
		if err != nil {
			return nil, err
		}
		efConstruction, err := nonNegativeIntFromProto("HNSW EFConstruction", p.Hnsw.EfConstruction)
		if err != nil {
			return nil, err
		}
		return xvec.HNSWIndexParams{Metric: xvec.MetricType(p.Hnsw.Metric), M: m, EFConstruction: efConstruction, Quantize: xvec.QuantizeType(p.Hnsw.Quantize), UseContiguousMemory: p.Hnsw.UseContiguousMemory, Quantizer: quantizerFromProto(p.Hnsw.Quantizer)}, nil
	case *xvecv1.IndexParams_IvfRabitq:
		if p.IvfRabitq == nil {
			return nil, invalid("nil IVF RaBitQ index params")
		}
		nlist, err := nonNegativeIntFromProto("IVF RaBitQ NList", p.IvfRabitq.Nlist)
		if err != nil {
			return nil, err
		}
		totalBits, err := nonNegativeIntFromProto("IVF RaBitQ TotalBits", p.IvfRabitq.TotalBits)
		if err != nil {
			return nil, err
		}
		sampleCount, err := nonNegativeIntFromProto("IVF RaBitQ SampleCount", p.IvfRabitq.SampleCount)
		if err != nil {
			return nil, err
		}
		return xvec.IVFRaBitQIndexParams{Metric: xvec.MetricType(p.IvfRabitq.Metric), NList: nlist, TotalBits: totalBits, SampleCount: sampleCount}, nil
	case *xvecv1.IndexParams_Ivf:
		if p.Ivf == nil {
			return nil, invalid("nil IVF index params")
		}
		nlist, err := nonNegativeIntFromProto("IVF NList", p.Ivf.Nlist)
		if err != nil {
			return nil, err
		}
		niterations, err := nonNegativeIntFromProto("IVF NIterations", p.Ivf.Niterations)
		if err != nil {
			return nil, err
		}
		return xvec.IVFIndexParams{Metric: xvec.MetricType(p.Ivf.Metric), NList: nlist, NIterations: niterations, UseSOAR: p.Ivf.UseSoar, Quantize: xvec.QuantizeType(p.Ivf.Quantize), Quantizer: quantizerFromProto(p.Ivf.Quantizer)}, nil
	case *xvecv1.IndexParams_Diskann:
		if p.Diskann == nil {
			return nil, invalid("nil DiskANN index params")
		}
		maxDegree, err := nonNegativeIntFromProto("DiskANN MaxDegree", p.Diskann.MaxDegree)
		if err != nil {
			return nil, err
		}
		listSize, err := nonNegativeIntFromProto("DiskANN ListSize", p.Diskann.ListSize)
		if err != nil {
			return nil, err
		}
		pqChunks, err := nonNegativeIntFromProto("DiskANN PQChunks", p.Diskann.PqChunks)
		if err != nil {
			return nil, err
		}
		return xvec.DiskANNIndexParams{Metric: xvec.MetricType(p.Diskann.Metric), MaxDegree: maxDegree, ListSize: listSize, PQChunks: pqChunks, Quantize: xvec.QuantizeType(p.Diskann.Quantize), Quantizer: quantizerFromProto(p.Diskann.Quantizer)}, nil
	case *xvecv1.IndexParams_Vamana:
		if p.Vamana == nil {
			return nil, invalid("nil Vamana index params")
		}
		maxDegree, err := nonNegativeIntFromProto("Vamana MaxDegree", p.Vamana.MaxDegree)
		if err != nil {
			return nil, err
		}
		searchListSize, err := nonNegativeIntFromProto("Vamana SearchListSize", p.Vamana.SearchListSize)
		if err != nil {
			return nil, err
		}
		maxOcclusionSize, err := nonNegativeIntFromProto("Vamana MaxOcclusionSize", p.Vamana.MaxOcclusionSize)
		if err != nil {
			return nil, err
		}
		return xvec.VamanaIndexParams{Metric: xvec.MetricType(p.Vamana.Metric), MaxDegree: maxDegree, SearchListSize: searchListSize, Alpha: p.Vamana.Alpha, MaxOcclusionSize: maxOcclusionSize, SaturateGraph: p.Vamana.SaturateGraph, UseContiguousMemory: p.Vamana.UseContiguousMemory, UseIDMap: p.Vamana.UseIdMap, Quantize: xvec.QuantizeType(p.Vamana.Quantize), Quantizer: quantizerFromProto(p.Vamana.Quantizer)}, nil
	case *xvecv1.IndexParams_Fts:
		if p.Fts == nil {
			return nil, invalid("nil FTS index params")
		}
		if err := rejectRemoteFTSPaths(p.Fts.ExtraParams); err != nil {
			return nil, err
		}
		return xvec.FTSIndexParams{Tokenizer: p.Fts.Tokenizer, Filters: append([]string(nil), p.Fts.Filters...), ExtraParams: p.Fts.ExtraParams}, nil
	default:
		return nil, invalid("index params kind is missing")
	}
}

func rejectRemoteFTSPaths(extraParams string) error {
	if extraParams == "" {
		return nil
	}
	var extra map[string]json.RawMessage
	if err := json.Unmarshal([]byte(extraParams), &extra); err != nil {
		return nil
	}
	for _, name := range []string{"jieba_dict_dir", "user_dict_path"} {
		if _, exists := extra[name]; exists {
			return invalid("remote FTS parameter %q is not supported", name)
		}
	}
	return nil
}

func queryOptionsToProto(o xvec.QueryOptions) *xvecv1.QueryOptions {
	return &xvecv1.QueryOptions{Radius: o.Radius, Linear: o.Linear, UseRefiner: o.UseRefiner}
}
func queryOptionsFromProto(o *xvecv1.QueryOptions) xvec.QueryOptions {
	if o == nil {
		return xvec.QueryOptions{}
	}
	return xvec.QueryOptions{Radius: o.Radius, Linear: o.Linear, UseRefiner: o.UseRefiner}
}

func QueryParamsToProto(params xvec.QueryParams) (*xvecv1.QueryParams, error) {
	if params == nil {
		return nil, nil
	}
	switch p := params.(type) {
	case xvec.FlatQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_Flat{Flat: &xvecv1.FlatQueryParams{Options: queryOptionsToProto(p.QueryOptions), ScaleFactor: p.ScaleFactor}}}, nil
	case *xvec.FlatQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	case xvec.HNSWQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_Hnsw{Hnsw: &xvecv1.HNSWQueryParams{Options: queryOptionsToProto(p.QueryOptions), Ef: int64(p.EF), PrefetchOffset: p.PrefetchOffset, PrefetchLines: p.PrefetchLines}}}, nil
	case *xvec.HNSWQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	case xvec.IVFRaBitQQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_IvfRabitq{IvfRabitq: &xvecv1.IVFRaBitQQueryParams{Options: queryOptionsToProto(p.QueryOptions), Nprobe: int64(p.NProbe), ScaleFactor: p.ScaleFactor}}}, nil
	case *xvec.IVFRaBitQQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	case xvec.IVFQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_Ivf{Ivf: &xvecv1.IVFQueryParams{Options: queryOptionsToProto(p.QueryOptions), Nprobe: int64(p.NProbe), ScaleFactor: p.ScaleFactor}}}, nil
	case *xvec.IVFQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	case xvec.DiskANNQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_Diskann{Diskann: &xvecv1.DiskANNQueryParams{Options: queryOptionsToProto(p.QueryOptions), ListSize: int64(p.ListSize)}}}, nil
	case *xvec.DiskANNQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	case xvec.VamanaQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_Vamana{Vamana: &xvecv1.VamanaQueryParams{Options: queryOptionsToProto(p.QueryOptions), EfSearch: int64(p.EFSearch), PrefetchOffset: p.PrefetchOffset, PrefetchLines: p.PrefetchLines}}}, nil
	case *xvec.VamanaQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	case xvec.FTSQueryParams:
		return &xvecv1.QueryParams{Kind: &xvecv1.QueryParams_Fts{Fts: &xvecv1.FTSQueryParams{DefaultOperator: p.DefaultOperator}}}, nil
	case *xvec.FTSQueryParams:
		if p != nil {
			return QueryParamsToProto(*p)
		}
	default:
		return nil, invalid("unsupported query params %T", params)
	}
	return nil, invalid("unsupported query params %T", params)
}

func QueryParamsFromProto(message *xvecv1.QueryParams) (xvec.QueryParams, error) {
	if message == nil {
		return nil, nil
	}
	switch p := message.Kind.(type) {
	case *xvecv1.QueryParams_Flat:
		if p.Flat != nil {
			return xvec.FlatQueryParams{QueryOptions: queryOptionsFromProto(p.Flat.Options), ScaleFactor: p.Flat.ScaleFactor}, nil
		}
	case *xvecv1.QueryParams_Hnsw:
		if p.Hnsw != nil {
			ef, err := nonNegativeIntFromProto("HNSW EF", p.Hnsw.Ef)
			if err != nil {
				return nil, err
			}
			return xvec.HNSWQueryParams{QueryOptions: queryOptionsFromProto(p.Hnsw.Options), EF: ef, PrefetchOffset: p.Hnsw.PrefetchOffset, PrefetchLines: p.Hnsw.PrefetchLines}, nil
		}
	case *xvecv1.QueryParams_IvfRabitq:
		if p.IvfRabitq != nil {
			nprobe, err := nonNegativeIntFromProto("IVF RaBitQ NProbe", p.IvfRabitq.Nprobe)
			if err != nil {
				return nil, err
			}
			return xvec.IVFRaBitQQueryParams{QueryOptions: queryOptionsFromProto(p.IvfRabitq.Options), NProbe: nprobe, ScaleFactor: p.IvfRabitq.ScaleFactor}, nil
		}
	case *xvecv1.QueryParams_Ivf:
		if p.Ivf != nil {
			nprobe, err := nonNegativeIntFromProto("IVF NProbe", p.Ivf.Nprobe)
			if err != nil {
				return nil, err
			}
			return xvec.IVFQueryParams{QueryOptions: queryOptionsFromProto(p.Ivf.Options), NProbe: nprobe, ScaleFactor: p.Ivf.ScaleFactor}, nil
		}
	case *xvecv1.QueryParams_Diskann:
		if p.Diskann != nil {
			listSize, err := nonNegativeIntFromProto("DiskANN ListSize", p.Diskann.ListSize)
			if err != nil {
				return nil, err
			}
			return xvec.DiskANNQueryParams{QueryOptions: queryOptionsFromProto(p.Diskann.Options), ListSize: listSize}, nil
		}
	case *xvecv1.QueryParams_Vamana:
		if p.Vamana != nil {
			efSearch, err := nonNegativeIntFromProto("Vamana EFSearch", p.Vamana.EfSearch)
			if err != nil {
				return nil, err
			}
			return xvec.VamanaQueryParams{QueryOptions: queryOptionsFromProto(p.Vamana.Options), EFSearch: efSearch, PrefetchOffset: p.Vamana.PrefetchOffset, PrefetchLines: p.Vamana.PrefetchLines}, nil
		}
	case *xvecv1.QueryParams_Fts:
		if p.Fts != nil {
			return xvec.FTSQueryParams{DefaultOperator: p.Fts.DefaultOperator}, nil
		}
	}
	return nil, invalid("query params kind is missing")
}
