<script setup>
const baseUrl = import.meta.env.BASE_URL
const paper = { title:'H3-World:', subtitle:'Turning Language Understanding into World Control', summary:'H3-World enables character and camera control across diverse visual environments.', venue:'Interactive World Models' }
const authors = [{name:'Danze Chen',mark:'1,2,♣'},{name:'Zeqing Wang',mark:'1,2,♣'},{name:'Ziyue Lin',mark:'3'},{name:'Xingyi Yang',mark:'3,*'},{name:'Yeying Jin',mark:'1,2,*,♦'}]
const links = [{label:'Project',icon:'↗',href:'https://github.com/Danzer1xxxxChan/H3-World'},{label:'GitHub',icon:'⌘',href:'https://github.com/Danzer1xxxxChan/H3-World'},{label:'Hugging Face',icon:'☺',href:'https://huggingface.co/DANNY621/H3-World'}]
</script>
<template><main>
  <header class="hero page-width"><div class="venue-pill">{{paper.venue}} · PROJECT PAGE</div><h1><span>{{paper.title}}</span>{{paper.subtitle}}</h1><p class="summary">{{paper.summary}}</p><div class="authors"><a v-for="author in authors" :key="author.name" href="#top">{{author.name}}<sup>{{author.mark}}</sup></a></div><div class="affiliations"><span><sup>1</sup> Tencent</span><span><sup>2</sup> National University of Singapore</span><span><sup>3</sup> The Hong Kong Polytechnic University</span></div><div class="author-notes"><span>♣ Research internship at Tencent</span><span>* Corresponding authors</span><span>♦ Project leader</span></div><nav class="link-row" aria-label="Publication links"><a v-for="link in links" :key="link.label" :href="link.href" target="_blank" rel="noopener noreferrer"><span>{{link.icon}}</span>{{link.label}}</a></nav></header>
  <section class="page-width hero-media" aria-label="Paper teaser"><figure><img :src="baseUrl + 'h3-world-teaser.jpg'" alt="H3-World teaser showing diverse visual environments"><figcaption><strong>H3-World</strong> enables character and camera control across diverse visual environments.</figcaption></figure></section>
</main></template>
