const LATEXJS_CDN = 'https://cdn.jsdelivr.net/npm/latex.js@0.12.6/dist/'

export async function downloadLatexAsPdf(latex: string): Promise<void> {
  const printWin = window.open('', '_blank')
  if (!printWin) {
    // Popup blocked — fall back to .tex file download
    downloadTex(latex)
    return
  }

  try {
    const { parse, HtmlGenerator } = await import('latex.js')
    const generator = new HtmlGenerator({ hyphenate: false })
    parse(latex, { generator })

    const head = printWin.document.head
    const body = printWin.document.body

    const styles = generator.stylesAndScripts(LATEXJS_CDN)
    head.appendChild(printWin.document.adoptNode(styles))
    body.appendChild(printWin.document.adoptNode(generator.domFragment()))

    // Extra print CSS so the page renders cleanly as a document
    const printStyle = printWin.document.createElement('style')
    printStyle.textContent = `
      body { font-family: serif; font-size: 12pt; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 2cm; }
      @media print { body { margin: 0; padding: 0; } }
    `
    head.appendChild(printStyle)

    printWin.document.close()
    setTimeout(() => {
      printWin.focus()
      printWin.print()
    }, 800)
  } catch {
    printWin.close()
    downloadTex(latex)
  }
}

function downloadTex(latex: string): void {
  const blob = new Blob([latex], { type: 'application/x-tex;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = 'rapport-conformite.tex'
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
