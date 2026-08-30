import { sidebar } from "vuepress-theme-hope";

export default sidebar({
  "/": [
    "",
    {
      text: "User guides",
      children: [
        "collection",
        "query",
        "multi-query",
        "full-text-search",
        "runtime",
      ],
    },
    {
      text: "Maintainer guides",
      children: ["vector-indexes", "storage"],
    },
  ],
});
