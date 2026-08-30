import { navbar } from "vuepress-theme-hope";

export default navbar([
  { text: "Home", link: "/" },
  { text: "Collections", link: "/collection.html" },
  { text: "Queries", link: "/query.html" },
  { text: "Full-text search", link: "/full-text-search.html" },
  {
    text: "Reference",
    children: [
      { text: "Go API", link: "https://pkg.go.dev/github.com/gorse-io/xvec" },
      { text: "GitHub", link: "https://github.com/gorse-io/xvec" },
    ],
  },
]);
