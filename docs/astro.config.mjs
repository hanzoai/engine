import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

// Deployed at https://hanzoai.github.io/engine/
// Adjust `site` + `base` if we move to docs.hanzo.
export default defineConfig({
  site: 'https://hanzoai.github.io',
  base: '/hanzo',
  integrations: [
    starlight({
      title: 'hanzo',
      description: 'Fast, flexible LLM inference engine written in Rust.',
      social: [
        { icon: 'github', label: 'GitHub', href: 'https://github.com/hanzoai/engine' },
        { icon: 'discord', label: 'Discord', href: 'https://discord.gg/SZrecqK8qw' },
      ],
      editLink: {
        baseUrl: 'https://github.com/hanzoai/engine/edit/master/docs/',
      },
      sidebar: [
        {
          label: 'Start here',
          slug: 'start-here',
        },
        {
          label: 'Tutorials',
          items: [{ autogenerate: { directory: 'tutorials' } }],
        },
        {
          label: 'Guides',
          items: [{ autogenerate: { directory: 'guides' } }],
        },
        {
          label: 'Reference',
          items: [{ autogenerate: { directory: 'reference' } }],
        },
        {
          label: 'Explanation',
          items: [{ autogenerate: { directory: 'explanation' } }],
        },
      ],
      components: {
        // Measured claims are entry points in the footer, not body copy.
        Footer: './src/components/Footer.astro',
      },
      customCss: ['./src/styles/custom.css'],
    }),
  ],
});
