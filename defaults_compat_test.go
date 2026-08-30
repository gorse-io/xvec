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

package xvec

import (
	"encoding/json"
	"os"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestPublicDefaultsCompatibility(t *testing.T) {
	data, err := os.ReadFile("testdata/cpp_defaults_58375ff.json")
	require.NoError(t, err)

	var fixture struct {
		Baseline string                    `json:"baseline_commit"`
		Defaults map[string]map[string]any `json:"defaults"`
	}
	{
		err := json.Unmarshal(data, &fixture)
		require.NoError(t, err)
	}
	require.True(t, fixture.Baseline == "58375ff7b8fdd0d6fc7d234e47567b179777883b")
	delete(fixture.Defaults, "hnsw_rabitq_index")
	fixture.Defaults["ivf_rabitq_index"] = map[string]any{
		"n_list": float64(1024), "total_bits": float64(7), "sample_count": float64(0),
	}
	delete(fixture.Defaults, "hnsw_rabitq_query")
	fixture.Defaults["ivf_rabitq_query"] = map[string]any{
		"nprobe": float64(10), "scale_factor": float64(10),
	}

	invert := NewInvertIndexParams()
	hnsw := NewHNSWIndexParams(MetricTypeL2)
	rabitq := NewIVFRaBitQIndexParams(MetricTypeL2)
	ivf := NewIVFIndexParams(MetricTypeL2)
	diskANN := NewDiskANNIndexParams(MetricTypeL2)
	vamana := NewVamanaIndexParams(MetricTypeL2)
	fts := NewFTSIndexParams()
	flatQuery := NewFlatQueryParams()
	hnswQuery := NewHNSWQueryParams()
	rabitqQuery := NewIVFRaBitQQueryParams()
	ivfQuery := NewIVFQueryParams()
	diskANNQuery := NewDiskANNQueryParams()
	vamanaQuery := NewVamanaQueryParams()
	ftsQuery := NewFTSQueryParams()
	collection := NewCollectionSchema("abc", NewField("id", DataTypeString))

	got := map[string]map[string]any{
		"invert_index":     {"range": invert.EnableRangeOptimization, "extended_wildcard": invert.EnableExtendedWildcard},
		"hnsw_index":       {"m": hnsw.M, "ef_construction": hnsw.EFConstruction},
		"ivf_rabitq_index": {"n_list": rabitq.NList, "total_bits": rabitq.TotalBits, "sample_count": rabitq.SampleCount},
		"ivf_index":        {"n_list": ivf.NList, "n_iterations": ivf.NIterations, "use_soar": ivf.UseSOAR},
		"diskann_index":    {"max_degree": diskANN.MaxDegree, "list_size": diskANN.ListSize, "pq_chunks": diskANN.PQChunks},
		"vamana_index":     {"max_degree": vamana.MaxDegree, "search_list_size": vamana.SearchListSize, "alpha": vamana.Alpha, "saturate_graph": vamana.SaturateGraph},
		"fts_index":        {"tokenizer": fts.Tokenizer, "filters": fts.Filters, "extra_params": fts.ExtraParams},
		"flat_query":       {"scale_factor": flatQuery.ScaleFactor},
		"hnsw_query":       {"ef": hnswQuery.EF, "prefetch_offset": hnswQuery.PrefetchOffset, "prefetch_lines": hnswQuery.PrefetchLines},
		"ivf_rabitq_query": {"nprobe": rabitqQuery.NProbe, "scale_factor": rabitqQuery.ScaleFactor},
		"ivf_query":        {"nprobe": ivfQuery.NProbe, "scale_factor": ivfQuery.ScaleFactor},
		"diskann_query":    {"list_size": diskANNQuery.ListSize},
		"vamana_query":     {"ef_search": vamanaQuery.EFSearch, "prefetch_offset": vamanaQuery.PrefetchOffset, "prefetch_lines": vamanaQuery.PrefetchLines},
		"fts_query":        {"default_operator": ftsQuery.DefaultOperator},
		"collection":       {"max_docs_per_segment": collection.MaxDocsPerSegment},
	}
	wantJSON, _ := json.Marshal(fixture.Defaults)
	gotJSON, _ := json.Marshal(got)
	require.Equal(t, string(wantJSON), string(gotJSON))
}
