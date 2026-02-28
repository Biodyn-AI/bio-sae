export default function About() {
  return (
    <div className="max-w-3xl mx-auto px-4 py-12 space-y-10">
      <section className="space-y-4">
        <h1 className="text-3xl font-bold text-white">About the Author</h1>
        <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-4">
          <h2 className="text-xl font-semibold text-gray-100">Ihor Kendiukhov</h2>
          <p className="text-gray-400 leading-relaxed">
            Researcher at the intersection of computational biology, mechanistic interpretability,
            and foundation models for single-cell genomics. This atlas presents a comprehensive
            Sparse Autoencoder decomposition of{' '}
            <span className="text-gray-200">scGPT whole-human (33M cells)</span>, revealing
            biological features across all 12 transformer layers.
          </p>
          <div className="flex flex-wrap gap-4 pt-2">
            <a
              href="https://github.com/Kendiukhov"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-gray-200 transition-colors"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z" />
              </svg>
              GitHub
            </a>
            <a
              href="https://www.linkedin.com/in/ikendiukhov/"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2.5 bg-blue-900/30 hover:bg-blue-900/50 border border-blue-800/50 rounded-lg text-blue-300 transition-colors"
            >
              <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24">
                <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z" />
              </svg>
              LinkedIn
            </a>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-200">About the Atlas</h2>
        <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-3 text-gray-400 leading-relaxed">
          <p>
            This interactive atlas presents the results of training <strong className="text-gray-200">TopK Sparse Autoencoders</strong>{' '}
            (d=512, 4x expansion to 2,048 features, k=32) on the residual stream activations of{' '}
            <strong className="text-gray-200">scGPT whole-human</strong> across all 12 transformer layers,
            using 3.56 million token positions from 3,000 Tabula Sapiens cells (immune, kidney, lung).
          </p>
          <p>
            Each feature represents a learned direction in the model's activation space that
            corresponds to a specific biological concept — from individual gene programs to
            complex pathway interactions. Features are annotated against five ontology databases:
          </p>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li><span className="text-blue-400">Gene Ontology (GO)</span> — Biological Process terms</li>
            <li><span className="text-green-400">KEGG</span> — Metabolic and signaling pathways</li>
            <li><span className="text-purple-400">Reactome</span> — Curated pathway processes</li>
            <li><span className="text-orange-400">STRING</span> — Protein-protein interaction network edges</li>
            <li><span className="text-red-400">TRRUST</span> — Transcription factor regulatory targets</li>
          </ul>
          <p>
            The scGPT whole-human model was trained on 33 million cells across all tissue types,
            using a 12-layer transformer with d_model=512. This atlas enables comparison with
            the companion <a href="https://biodyn-ai.github.io/geneformer-atlas/" className="text-blue-400 hover:underline">Geneformer Feature Atlas</a>{' '}
            (18 layers, d=1152, 4608 features/layer) to understand how different foundation model
            architectures represent biological knowledge.
          </p>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-200">Project</h2>
        <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-3 text-gray-400 leading-relaxed">
          <p>
            This atlas is part of the <a href="https://biodynai.com" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">BioDyn AI project</a>, focused on mechanistic interpretability for biological foundation models.
          </p>
          <p>
            Main project website:{' '}
            <a href="https://biodynai.com" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
              biodynai.com
            </a>
          </p>
          <p>Atlas websites in this project:</p>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>
              <a href="https://biodyn-ai.github.io/geneformer-atlas/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                Geneformer Atlas
              </a>
            </li>
            <li>
              <a href="https://biodyn-ai.github.io/scgpt-atlas/" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">
                scGPT Atlas
              </a>
            </li>
          </ul>
        </div>
      </section>

      <section className="space-y-4">
        <h2 className="text-xl font-semibold text-gray-200">Methods</h2>
        <div className="bg-gray-900/50 rounded-xl border border-gray-800 p-6 space-y-3 text-gray-400 leading-relaxed">
          <p>
            <strong className="text-gray-200">Activation extraction:</strong> Residual stream
            activations were extracted from all 12 layers of scGPT whole-human processing
            3,000 Tabula Sapiens cells (3.56M positions per layer, ~82 GB total).
          </p>
          <p>
            <strong className="text-gray-200">SAE training:</strong> TopK Sparse Autoencoders
            with k=32 active features per input were trained independently on each layer.
            The 4x expansion ratio (512 → 2,048 features) captures fine-grained biological
            representations.
          </p>
          <p>
            <strong className="text-gray-200">Annotation:</strong> Each feature's top-activating
            genes were tested for enrichment against GO, KEGG, Reactome, STRING, and TRRUST
            databases using Fisher's exact test with BH correction (p &lt; 0.05).
          </p>
          <p>
            <strong className="text-gray-200">Causal validation:</strong> Activation patching
            on Layer 7 features confirmed that individual features causally drive
            specific gene logits.
          </p>
        </div>
      </section>
    </div>
  )
}
