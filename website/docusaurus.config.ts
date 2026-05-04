import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// Math plugins (LaTeX)
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'Ames Housing — ML Pipeline',
  tagline: 'Pipeline end-to-end per la previsione dei prezzi immobiliari sul dataset Ames (De Cock 2011)',
  favicon: 'img/favicon.ico',

  future: {
    v4: true,
  },

  url: 'https://fedcal.github.io',
  baseUrl: '/ames-housing-price-pipeline/',

  organizationName: 'fedcal',
  projectName: 'ames-housing-price-pipeline',

  onBrokenLinks: 'warn',
  onBrokenMarkdownLinks: 'warn',

  i18n: {
    defaultLocale: 'it',
    locales: ['it'],
  },

  markdown: {
    mermaid: true,
  },
  themes: ['@docusaurus/theme-mermaid'],

  stylesheets: [
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV',
      crossorigin: 'anonymous',
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/fedcal/ames-housing-price-pipeline/tree/main/website/',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
          showLastUpdateTime: true,
          showLastUpdateAuthor: false,
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.5,
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/social-card.png',
    metadata: [
      {name: 'keywords', content: 'machine learning, regression, ames housing, scikit-learn, xgboost, ridge, random forest, kaggle, tabular ml, pipeline'},
      {name: 'description', content: 'Pipeline ML end-to-end per la previsione dei prezzi immobiliari sul dataset Ames Housing. Ridge, Random Forest, XGBoost, tuning K-fold.'},
    ],
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Ames Housing',
      logo: {
        alt: 'Ames Housing Pipeline logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'mainSidebar',
          position: 'left',
          label: 'Documentazione',
        },
        {
          href: 'https://github.com/fedcal/ames-housing-price-pipeline',
          label: 'GitHub',
          position: 'right',
        },
        {
          href: 'https://federicocalo.dev',
          label: 'federicocalo.dev',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Documentazione',
          items: [
            {label: 'Introduzione', to: '/docs/intro'},
            {label: 'Teoria', to: '/docs/category/teoria'},
            {label: 'Scelte tecniche', to: '/docs/category/scelte-tecniche'},
          ],
        },
        {
          title: 'Codice',
          items: [
            {label: 'GitHub repo', href: 'https://github.com/fedcal/ames-housing-price-pipeline'},
            {label: 'Issues', href: 'https://github.com/fedcal/ames-housing-price-pipeline/issues'},
            {label: 'License (MIT)', href: 'https://github.com/fedcal/ames-housing-price-pipeline/blob/main/LICENSE'},
          ],
        },
        {
          title: 'Autore',
          items: [
            {label: 'federicocalo.dev', href: 'https://federicocalo.dev'},
            {label: 'GitHub', href: 'https://github.com/fedcal'},
            {label: 'LinkedIn', href: 'https://www.linkedin.com/in/federicocalo/'},
          ],
        },
      ],
      copyright: `© ${new Date().getFullYear()} Federico Calò · Progetto realizzato come parte del percorso <a href="https://datamasters.it/" target="_blank" rel="noopener noreferrer" style="color:#fff">Machine Learning Engineer di DataMasters/Skiller</a> · Distribuito sotto licenza MIT · <a href="https://federicocalo.dev" target="_blank" rel="noopener noreferrer" style="color:#fff"><strong>federicocalo.dev</strong></a>`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'python', 'yaml', 'json', 'toml', 'diff'],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: false,
      },
    },
    tableOfContents: {
      minHeadingLevel: 2,
      maxHeadingLevel: 4,
    },
    announcementBar: {
      id: 'portfolio-banner',
      content:
        '🔗 Esplora gli altri progetti su <a target="_blank" rel="noopener noreferrer" href="https://federicocalo.dev"><strong>federicocalo.dev</strong></a>',
      backgroundColor: '#4f46e5',
      textColor: '#ffffff',
      isCloseable: true,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
