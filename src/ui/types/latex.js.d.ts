declare module 'latex.js' {
  export class HtmlGenerator {
    constructor(options?: { hyphenate?: boolean; languagePatterns?: unknown })
    stylesAndScripts(baseURL: string): DocumentFragment
    domFragment(): DocumentFragment
    htmlDocument(): Document
  }

  export function parse(
    latex: string,
    options: { generator: HtmlGenerator }
  ): HtmlGenerator
}
