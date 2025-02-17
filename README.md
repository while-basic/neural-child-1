# Neural Child Development Dashboard

A sophisticated monitoring system for tracking and visualizing the development of a neural child. Built with Next.js, Shadcn/UI, and Plotly.js.

## Features

- Real-time monitoring of emotional states
- Development stage tracking
- Warning system with visual indicators
- Interactive data visualization
- Cognitive development metrics
- Social skills monitoring
- Physical development tracking

## Tech Stack

- **Framework**: Next.js 14
- **UI Components**: Shadcn/UI
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Charts**: Plotly.js
- **Icons**: Lucide Icons
- **Animations**: Framer Motion

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/neural-child-dashboard.git
```

2. Install dependencies:
```bash
cd neural-child-dashboard
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/
├── app/
│   ├── layout.tsx
│   ├── page.tsx
│   └── globals.css
├── components/
│   └── ui/
│       ├── card.tsx
│       ├── progress.tsx
│       └── tabs.tsx
├── lib/
│   ├── store.ts
│   └── utils.ts
└── types/
    └── index.ts
```

## Development

- **Components**: UI components are built using Shadcn/UI with custom styling
- **State Management**: Global state is managed using Zustand
- **Styling**: Custom Tailwind CSS classes with gradient borders and animations
- **Data Visualization**: Interactive charts using Plotly.js
- **Theme**: Dark theme with gradient accents and glowing effects

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Shadcn/UI for the beautiful component library
- Plotly.js for the powerful charting capabilities
- The Next.js team for the amazing framework
