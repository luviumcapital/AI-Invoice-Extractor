# FlexYield

Wardrobe ROI, anti-counterfeit vault, and social drip feed — track the true cost-per-wear of your clothes, verify authenticity, and export shareable "Flex Cards."

Originally generated in [v0.app](https://v0.app) and developed further here.

## Stack

- [Next.js](https://nextjs.org) (App Router) + TypeScript
- [Tailwind CSS v4](https://tailwindcss.com)
- [Framer Motion](https://motion.dev) for micro-interactions
- [lucide-react](https://lucide.dev) icons
- [html2canvas](https://html2canvas.hertzen.com) for Flex Card export
- [@vercel/analytics](https://vercel.com/docs/analytics) (production only)
- `localStorage` for client-side persistence — no backend or API keys required

## Getting started

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Scripts

- `npm run dev` — start the dev server
- `npm run build` — production build
- `npm run start` — run the production build
- `npm run lint` — lint the project

## Deployment

Designed for zero-config deployment on [Vercel](https://vercel.com)'s free tier. No environment variables or external services are required — all data lives in the browser via `localStorage`.
