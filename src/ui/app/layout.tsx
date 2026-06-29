import type { Metadata } from 'next'
import { JetBrains_Mono } from 'next/font/google'
import './globals.css'

const mono = JetBrains_Mono({
  subsets: ['latin'],
  weight: ['400', '700'],
})

export const metadata: Metadata = {
  title: 'compliance assistant',
  description: 'EU regulation compliance assistant — MiCA, DORA, AI Act, GDPR',
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="fr">
      <body className={`${mono.className} h-full`}>{children}</body>
    </html>
  )
}
