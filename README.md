# Academic Paper Project Page

一个基于 Vue 3 + Vite 的单页论文项目网站，参考 `academic-project-page-template-vue` 的内容结构重新设计。

## 本地运行

```bash
npm install
npm run dev
```

主要内容位于 `src/App.vue`，视觉样式位于 `src/style.css`。替换论文标题、作者、机构、摘要、资源链接、实验结果和 BibTeX 即可使用。

## GitHub Pages

项目已包含自动发布工作流。推送至 `RobinLin2002/paper-project-page` 的 `main` 分支后，在仓库 Settings → Pages 中将 Source 设为 GitHub Actions，网站会发布到：

`https://robinlin2002.github.io/paper-project-page/`
