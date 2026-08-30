import { defineUserConfig } from "vuepress";

import theme from "./theme.js";

export default defineUserConfig({
  base: "/",
  lang: "en-US",
  title: "xvec",
  description: "A pure-Go embedded vector database",
  theme,
});
