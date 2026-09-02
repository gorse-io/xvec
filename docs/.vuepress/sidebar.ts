import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/benchmark/": [
    "",
    { text: "Flat Index", link: "/benchmark/flat-index" },
    { text: "HNSW Index", link: "/benchmark/hnsw-index" },
    { text: "IVF Index", link: "/benchmark/ivf-index" },
    { text: "DiskANN Index", link: "/benchmark/diskann-index" },
  ],
  "/": [""],
});
