package main
import (
 "encoding/json"
 "fmt"
 "os"
 "io"
 "path/filepath"
 "github.com/parquet-go/parquet-go"
)
type Label struct {ID int64 `parquet:"id"`; Value string `parquet:"labels"`}
type Neighbor struct {ID int64 `parquet:"id"`; IDs []int64 `parquet:"neighbors_id,list"`}
func main() {
 root:=os.Args[1]
 labels,err:=parquet.ReadFile[Label](filepath.Join(root,"scalar_labels.parquet"));if err!=nil {panic(err)}
 byID:=map[int64]string{};counts:=map[string]int{}
 for _,l:=range labels {if _,found:=byID[l.ID];found {panic("duplicate ID")};byID[l.ID]=l.Value;counts[l.Value]++}
 if len(byID)!=100000 {panic("unexpected label count")}
 files,err:=filepath.Glob(filepath.Join(root,"neighbors_labels_*.parquet"));if err!=nil {panic(err)}
 summaries:=map[string]any{}
 for _,file:=range files {
  rows:=readNeighbors(file)
  name:=filepath.Base(file); label:=name[len("neighbors_labels_"):len(name)-len(".parquet")]
  if len(rows)!=1000 {panic("unexpected query count")}
  minN,maxN:=100000,0
  for _,row:=range rows {if len(row.IDs)<minN {minN=len(row.IDs)};if len(row.IDs)>maxN {maxN=len(row.IDs)};for _,id:=range row.IDs {if byID[id]!=label {panic(fmt.Sprintf("%s query=%d neighbor=%d label=%s",name,row.ID,id,byID[id]))}}}
  summaries[label]=map[string]any{"matching_documents":counts[label],"query_count":len(rows),"min_neighbors":minN,"max_neighbors":maxN}
 }
 data,err:=json.MarshalIndent(map[string]any{"document_count":len(byID),"label_counts":counts,"ground_truth":summaries},"","  ");if err!=nil {panic(err)}
 fmt.Println(string(data))
}

func readNeighbors(path string) []Neighbor {
 f,err:=os.Open(path);if err!=nil {panic(err)};defer f.Close()
 r:=parquet.NewReader(f);defer r.Close();var result []Neighbor;buffer:=make([]parquet.Row,100)
 for {n,err:=r.ReadRows(buffer);for _,raw:=range buffer[:n] {var row Neighbor;for _,v:=range raw {if v.IsNull(){continue};if v.Column()==0 {row.ID=v.Int64()} else if v.Column()==1 {row.IDs=append(row.IDs,v.Int64())}};result=append(result,row)};if err==io.EOF {break};if err!=nil {panic(err)}}
 return result
}
