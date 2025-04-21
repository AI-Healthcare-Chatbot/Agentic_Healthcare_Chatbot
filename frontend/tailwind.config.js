/** @type {import('tailwindcss').Config} */
module.exports = {
    content: [
      "./src/**/*.{js,jsx,ts,tsx}",
      "./public/index.html",
    ],
    theme: {
      extend: {
        colors: {
          'healthcare-blue': '#3b82f6',
          'healthcare-blue-dark': '#1d4ed8',
          'healthcare-red': '#ef4444',
          'healthcare-green': '#10b981',
          'healthcare-purple': '#8b5cf6',
          'healthcare-gray': '#6b7280',
        },
        animation: {
          'bounce-slow': 'bounce 1.5s infinite',
        },
        fontFamily: {
          sans: [
            'Inter',
            'ui-sans-serif',
            'system-ui',
            '-apple-system',
            'BlinkMacSystemFont',
            '"Segoe UI"',
            'Roboto',
            '"Helvetica Neue"',
            'Arial',
            '"Noto Sans"',
            'sans-serif',
            '"Apple Color Emoji"',
            '"Segoe UI Emoji"',
            '"Segoe UI Symbol"',
            '"Noto Color Emoji"',
          ],
        },
      },
    },
    plugins: [
      require('@tailwindcss/forms'),
    ],
  }