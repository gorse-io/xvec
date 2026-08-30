import { hopeTheme } from "vuepress-theme-hope";

import navbar from "./navbar.js";
import sidebar from "./sidebar.js";

export default hopeTheme({
  repo: "gorse-io/xvec",
  docsDir: "docs",
  logo: "/logo.png",
  navbar,
  sidebar,
  editLink: true,
  lastUpdated: true,
  contributors: true,
  displayFooter: true,
  footer: "Apache License 2.0",
  copyright: "Copyright © xvec contributors",
  markdown: {
    gfm: true,
  },
  plugins: {
    slimsearch: false,
  },
});
