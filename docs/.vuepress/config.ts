import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",
  lang: "en-US",
  title: "xvec",
  description: "A pure-Go embedded vector database",
  head: [
    ["meta", { name: "theme-color", content: "#2f8f6f" }],
  ],
  theme,
});
