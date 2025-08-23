import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'

import '@fontsource/urbanist/400.css'
import '@fontsource/urbanist/600.css'
import '@fontsource/roboto/400.css'
import '@fontsource/roboto/700.css'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
