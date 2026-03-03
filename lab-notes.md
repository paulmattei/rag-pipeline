---
title: "What Building a RAG Pipeline for Confluence and Jira Actually Taught Me"
description: "I built a RAG pipeline for Jira and Confluence. Here's what I learned."
pubDate: 2026-03-16
tags: ["rag"]
draft: true
---

## Things to think about:

Sections I should have:

Chunking — first decision you hit, most underestimated
Hybrid search — the moment you realise pure dense embeddings aren't enough
Reranking — the "why is it still returning slightly wrong stuff" problem
Query rewriting — often the highest leverage, least obvious intervention
Prompt engineering — how you instruct the generator to use the context faithfully

### What test content should I use.

In this project I'll be using weaviate for the first time. I'll probably have questions
so I settled on the idea of using their docs. I'll be a actual user so I can also be a judge
of how useful the answers from the system are. In addition, their website has search built in
"powered by Weaviate and Kapa.ai" so I can compare how this system performs vs something no doubt
more sophisticaed. 

### Performance metrics 
I will use to guage performance of the different strategies and techniques:
- Recall
- Precision
- Latency
- Cost
+ plus my own subjective experience in how it helps me navigate learning weaviate.



### Chunking strategies:

- Fixed size (250-500 char) # Need to pick
- Fixed size with overlap
- Paragraph (split on new line) (Recurive Character Splitting - in this case /n)
- Paragraph min size (combine headings with paragraph)
- Sentence wise Semantic Chunking
- LLM Chunking
- LLM Chunking with added summary context prepended (Context Aware Chunking)

### Vector database:
- I'll use Weviate because I just used it in the deeplearning.ai RAG course I did, so why not.
- I am curious about pgvector, maybe I'll compare this next time, put a pin in it for now.

### Model parameters that can be twiddled
- top_p, top_k, repetition_penalty and temperature

### Which LLM...?

I'll use Claude, because I have API account already.

## Things to consider when choosing an LLM

Input tokens/Output tokens cost  
Context window  
Time to first token  
Cutoff date  

### Which Embedding model?
Just pick something on together.ai


### Prompt Engineering - things to consider

- Template / structure for:  
    System prompt  
    User prompt w/ injected Retrieved Content  

- In context learning. One shot / few shot learning - giving examples of good output.  
    Can even use RAG for this so learning examples can be dynamic based on what the current query is.  

- Encouraging Resoning. Tell the LLM to 'think aloud' and thay tokens between <scratchpad></scratchpad> are  
    considered a space for thinking and brainstorming and not part of the final answer  

- Chain of Thought - 'think step by step' - Some models are reasoning models. Some providers give access to reasoning tokens, others hide them.
    For these you don't need to ask them to think step by step as this behaviour is trained in.

Reasoning models actually struggle with in context learning - perform best with clear goals and strict formats

### Context window management?

Don't include thinking tokens in messages format object.


### Future work ideas

Look at https://github.com/VectifyAI/PageIndex  
Look at pgvector  
Query parsing - Prompt rewriting by LLM  
https://arxiv.org/abs/2409.00729  
Ragas library - https://docs.ragas.io/en/stable/  
LangFuse  
Agentic RAG  

-----

## Day 1: 2026-03-03

Setup Atlassian account
Used my Google id
https://mattei.atlassian.net

Used Claude to create setup guide: atlassian-setup-guide.md

Create API key.
Key: dev
Created .env
Stored key there

Test key:

# Test key on Confluence
(source .env && curl -s -u paul@mattei.me.uk:$ATLASSIAN_API_KEY \
  https://mattei.atlassian.net/wiki/rest/api/space | python3 -m json.tool)

I'm going to import some documents into Confluence.
I'll use the API first to get familiar but I see there is also a python
SDK - book marked here probably useful in future...https://github.com/atlassian-api/atlassian-python-api

So the hirarchy in confluence seems to be:

Spaces > Pages

Spaces:

https://support.atlassian.com/confluence-cloud/docs/create-a-space/

Spaces appear to have settings. There are some default configs available when creating a Space:
- Collaboration
- Knowledge base

Not sure what I need let me understand what's availble in the config:

https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-space/

https://developer.atlassian.com/cloud/confluence/rest/v2/api-group-space/#api-spaces-post (This creates a space)

Maybe the difference is in the templateKey?

Some notes from Claude:

Here's what's available for Confluence space creation via the API:

  templateKey Values

  The key insight from the docs is that template keys are instance-specific,
  so you should first fetch them dynamically:

  GET /rest/create-dialog/1.0/space-blueprint/list

  That said, common well-known keys are:

  ┌───────────────┬──────────────────────────────────────────────────────┐
  │   Template    │                     templateKey                      │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ Blank/Default │ Confluence.Templates.Blueprints.CreateSpace.createSp │
  │               │ aceForm                                              │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ Knowledge     │ Confluence.SpaceBlueprints.KnowledgeBase.dialogForm  │
  │ Base          │                                                      │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ Team/Collabor │ Confluence.SpaceBlueprints.Team.dialogForm           │
  │ ation         │                                                      │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ Software      │ Confluence.SpaceBlueprints.SoftwareProject.Dialog.di │
  │ Project       │ alogForm                                             │
  ├───────────────┼──────────────────────────────────────────────────────┤
  │ Personal      │ Confluence.Templates.Blueprints.CreateSpace.createPe │
  │               │ rsonalSpaceForm                                      │
  └───────────────┴──────────────────────────────────────────────────────┘

  POST /wiki/api/v2/spaces Body Parameters

  {
    "name": "My Space",
    "key": "MYSPACE",
    "alias": "my-space",
    "description": { "value": "...", "representation": "plain" },
    "templateKey": "Confluence.SpaceBlueprints.KnowledgeBase.dialogForm",
    "createPrivateSpace": false,
    "roleAssignments": []
  }

  Recommendation

  The "Collaboration" option you see in the UI likely maps to the Team
  blueprint (Team.dialogForm), while "Knowledge base" maps to KnowledgeBase
  (KnowledgeBase.dialogForm). But verify against your specific instance using
   the blueprint list endpoint first — Atlassian's own docs warn the keys can
   differ per instance.

  Sources:
  - How to create spaces with blueprints through the Confluence REST API
  - Confluence Cloud REST API v2 - Space


  I'll setup Bruno to explore the API.

  In the end decide against templates. Seems like a side quest, will just create basic pages.

  I will use the weaviate docs as this seems like a good use case. I'm learning weaviate so I can use what I build ans see how good it is.

  The pipeline will be:

  Weaviate github docs > some python script I need to build > confluence.

  What should the python script algorithm be?

Clone/pull the Weaviate docs repo from GitHub

The docs are written with Docusaurus. The content is
blogs
papers
quickstart guides
code

Docs repo code ts, jsx, html, css and yaml files files.

File count:
MD files: 3797
MDX files: 299
Py files: 32
Ts files: 23289
Jsx files: 751
Go files: 6
Java files: 0
Html files: 213
Css files: 1453
Json files: 4997
Yaml files: 106
Yml files: 358

## Day 2: 2026-03-04

All the document content for our RAG project is exclusivly in the md and mdx files.

Each peice of documention is not in a self contained file though, exploring the mdx files
we find there are references to other text that need to be resolved to get a complete document.

### Pattern 1: Direct MDX include
                                                                 
  A page imports another .mdx file and renders it as a component:

  import GenerativeFeedbackLoopsSearch from
  '/_includes/code/generative.feedback.loops.search.mdx';

  Some explanation text here...

  <GenerativeFeedbackLoopsSearch/>

  The imported file (generative.feedback.loops.search.mdx)
  contains tabbed code blocks:

  import Tabs from '@theme/Tabs';
  import TabItem from '@theme/TabItem';

  <Tabs groupId="languages">
  <TabItem value="py" label="Python">

  ```python
  descriptions = client.query\
      .get("Listing", generate_properties)\
      .with_generate(single_prompt=generatePrompt)\
      .with_limit(5)\
      .do()["data"]["Get"]["Listing"]
```

### Pattern 2: Raw code file + FilteredTextBlock

  A page imports a raw .py file and extracts a section using
  marker comments:

```mdx
  import FilteredTextBlock from
  '@site/src/components/Documentation/FilteredTextBlock';
  import EndToEndPyCode from
  '!!raw-loader!/_includes/code/quickstart/endtoend.py';

  <FilteredTextBlock
    text={EndToEndPyCode}
    startMarker="# ===== define collection ====="
    endMarker="# ===== import data ====="
    language="py"
  />
```
  The imported Python file uses marker comments to delineate
  extractable sections:

```py
  # ===== define collection =====
  questions = client.collections.create(
      name="Question",
      vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_
  openai(),
      generative_config=wvc.config.Configure.Generative.openai()
  )

  # ===== import data =====
  resp = requests.get('https://raw.githubusercontent.com/...')
```
  FilteredTextBlock slices between the two markers, so only the
  collection definition code ends up in the rendered page. This
  is why resolve_document in the import script has to handle both
  patterns.

Example:

Parent page:                                                   
  weaviate-docs/blog/2023-05-05-generative-feedback-loops/index.mdx
  (https://weaviate.io/blog/generative-feedback-loops-with-llms#generate-a-description-for-each-listing)                                                         
                  
  Pattern 1 — imported MDX snippets:                             
    - weaviate-docs/_includes/code/generative.feedback.loops.descri
    ption.mdx
    - weaviate-docs/_includes/code/generative.feedback.loops.search
    .mdx
    -
    weaviate-docs/_includes/code/generative.feedback.loops.loop.mdx

  Pattern 2 — FilteredTextBlock with raw code:
    - weaviate-docs/_includes/code/quickstart/collection.definition
    .mdx (the MDX that uses FilteredTextBlock)
    - weaviate-docs/_includes/code/quickstart/endtoend.py (the raw
    Python file with marker comments)
    - weaviate-docs/_includes/code/quickstart/endtoend.py3.py
    (Python v3 variant)
    - weaviate-docs/_includes/code/quickstart/endtoend.ts
    (TypeScript variant)
    - weaviate-docs/_includes/code/quickstart/endtoend-v2.ts
    (TypeScript v2 variant)
    -
    weaviate-docs/_includes/code/quickstart/go-collection-define.go
    (Go variant)

So we need a pipeline to resolve all these imports and build a set
of final documents. The pipeline at a high level looks like this:

![alt](./high-level-document-rendering.svg)

build_import_graph(repo_dir)
   ├── rglob("*") for .md/.mdx files         → 329 files
   ├── parse_file_imports() on each           → resolve import paths
   └── return ImportGraph                      → {file: [dependencies]}

graph.top_level_pages()
   └── files not imported by any other file   → ~295 top-level pages

To build the Document:
For each top-level page:
   resolve_document(page_path, repo_dir)
   ├── parse_document()                       → Document with raw body
   ├── inline_mdx_includes()                  → replace <Component /> with file content
   │   ├── parse_mdx_imports()                → {name: Path} for .md/.mdx imports
   │   ├── read each imported file
   │   ├── strip_frontmatter() on included content
   │   └── recurse into included content       → handle nested includes
   ├── resolve_filtered_text_blocks()          → replace <FilteredTextBlock /> with code
   │   ├── parse_raw_imports()                → {var: path} for !!raw-loader! imports
   │   ├── read source file
   │   └── extract_between_markers()          → pull code section
   └── strip_mdx_syntax()                     → remove remaining JSX, convert admonitions

An LLM architecture review identified several edge cases and imporvements that need to be made.
Documented in architecture.md - another side quest. I'll keep moving forward and come back to this
at a later date. We have cleaned documents now and I think it's good enough.

I've decided to jump forward and build the first RAG system from this cleaned data. We'll
experiment with Confluence at a later stage - this blog will now be "Building a RAG Pipeline Over Technical Documentation"

## Chat with Claude:

Take a look at the notes now. These are raw, will need to be editied into a blog. Critically review the progress I'm making.

13:07

Prepared to scrutinize project progress and editorial refinement needs
Prepared to scrutinize project progress and editorial refinement needs
Good progress on the hard part — you've done genuine engineering work figuring out Weaviate's Docusaurus import resolution. The two-pattern discovery (MDX includes vs FilteredTextBlock with marker comments) and the import graph approach is exactly the kind of real-world complexity that makes this blog post worth reading. Nobody writes about this because most RAG tutorials start with clean text files.

A few critical observations though:

You've skipped Confluence. The blog post is called "What Building a RAG Pipeline for Confluence and Jira Actually Taught Me" and you've just decided to bypass the Confluence write step entirely. I understand the impulse — you've got clean documents, you want to get to the RAG part. But loading into Confluence via the API was the whole point of demonstrating Atlassian API fluency. The markdown-to-storage-format conversion, dealing with code blocks in Confluence's HTML format, preserving page hierarchy — those are the enterprise integration challenges that differentiate this post. Right now you've built a Docusaurus doc parser, which is interesting but isn't the story you set out to tell.

I'd push back on "experiment with Confluence at a later stage." That later stage won't come — you'll be deep into chunking and retrieval and Confluence will feel like going backwards. Do it now while you have the clean documents ready to push. It doesn't need to be perfect — even a basic conversion that handles headings, paragraphs, and code blocks is enough. The API quirks you hit are the blog content.

Jira hasn't appeared at all yet. Same concern. It's in the title.

The "jump forward" instinct is risky. You've done the unsexy parsing work well. The temptation now is to rush to the fun part (chunking, embeddings, retrieval). But the value of this post is the end-to-end journey including the messy integration work. If you skip to chunking, you end up with a generic RAG tutorial built on markdown files — which is what every other RAG post already is.

What's working well: The import graph architecture is solid and well-documented. The decision to use Weaviate docs is paying off — you're already finding real complexity. The notebook format is capturing decisions and dead ends (template side-quest, architecture.md rabbit hole) naturally. The SVG diagram reference suggests you're thinking visually, which will help the final post.

My recommendation: Spend one more day getting the Confluence load working. Push your resolved documents into Confluence pages via the API. Then ingest from Confluence (not from the local markdown) into your RAG pipeline. That round-trip is the story. The chunking and retrieval work will be more interesting too, because you'll be dealing with Confluence storage format HTML rather than clean markdown — which is what an actual enterprise RAG pipeline faces.

It felt a bit manufactured, who's doing this. Taking a random set of documents, cleaning them and then importing into Confluence. Is this useful or very specific to where I've ended up? What's more valuable, gettting to RAG to data wrangling?

13:09

Weighed artificial import step against genuine enterprise RAG value
Weighed artificial import step against genuine enterprise RAG value
Honest answer: you're right, it is manufactured. Nobody takes docs from GitHub, pushes them into Confluence, then pulls them back out for RAG. That's a round-trip that exists to tick a box, not to solve a real problem.

But the question isn't "import vs RAG." It's about what the blog post actually demonstrates. And there are really two different posts hiding here.

The post you're naturally writing is "Building a RAG Pipeline Over Technical Documentation" — the Docusaurus parsing, the import graph, the chunking, the retrieval, the evaluation against Weaviate's own search. That's genuine and interesting. The Confluence detour would weaken it.

The post that demonstrates Atlassian API fluency is a different project — one where data already lives in Confluence and Jira (as it does in every enterprise) and you build a pipeline that reads from those APIs, handles storage format HTML and ADF, deals with the messiness, and produces good retrieval. That's a real problem and it's directly relevant to companies like Glean.

You're trying to force both into one post and it's creating the artificiality you're sensing.

My revised advice: drop the Confluence import step. Get to RAG. Build the pipeline over your cleaned Weaviate docs, evaluate it properly, write that up as blog post #1. It's already a strong post — the Docusaurus parsing alone is more interesting data wrangling than most RAG tutorials include.

Then for blog post #2, start from the other direction — data already in Confluence and Jira, build the ingestion layer that reads from those APIs, and run your RAG pipeline against genuinely enterprise-shaped content. That's where the Atlassian API knowledge lives naturally.

Two posts, each authentic. Better than one manufactured hybrid.

yeh agreed. One thing I've already learned. Cleaning data is not simple.

13:11

Validated user's insight on data cleaning complexity
Validated user's insight on data cleaning complexity
That's the single most valuable line in your notebook so far, and it's the insight that separates practitioners from tutorial-followers.

Every RAG tutorial starts with clean text. Nobody talks about the hours spent figuring out that Docusaurus uses two different import patterns, or that you need to build an import graph just to get a complete document, or that "clean enough" is a judgement call you make at some point because perfect parsing is a bottomless pit.

That's also the insight that resonates with enterprise buyers. When a customer asks "how hard is it to set up RAG over our internal docs?" — the honest answer is that the model and the vector database are the easy parts. The data wrangling is where projects actually stall. An SE who can say that from experience, with a blog post backing it up, is infinitely more credible than one who's only done it with clean datasets.

Put that line in the blog post. Probably near the top.

---

So on to chunking...

But before that, let me break this file up a bit.
Refactored out into models.py, parsing.py, and a main.py

Now creating a chunking.py

At the start of these notes we outlined a number of chunking strategies,
I'd like to implement several so we can compare - our main 'chunker' will have a 'stratgy' argument.
The first and most simple. 500-chars (no overlap)

And we're not even going to map document path to each chunk or anything more sophisticated than simply chunk, at this stage.

```py
def create_chunks_500_char(text):
    return [text[i:i+500] for i in range(0, len(text), 500)]
```

We can either feed these chunks into weaviate or we can
do the embeddings ourselves and pass these to weaviate. In the
future I'm keen to explore the practical difference between 
HNSW and bruteforce dot product so I'm going to do the embeddings
outside of weaviate and pass them in, it's simple enough.

Of the myriad of embedding model options I chose
BAAI/bge-base-en-v1.5 as it seems a popular choice and amoung the handful
I tested it ran pretty fast. It took < 5 mins to embed this corpus on my Macbook.

```py
for doc in documents:
    chunks = create_chunks(doc.body, strategy="500-char")
    all_the_chunks.extend(chunks)
```

```py
def embed_chunks(chunks):
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device="mps")
    embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True)
    return embeddings

```py
with collection.batch.fixed_size(batch_size=1000) as batch:
    for chunk, embedding in zip(all_the_chunks, document_embeddings):
        batch.add_object(properties={"text": chunk}, vector=embedding.tolist())
```

Now the chunks are loaded into weaviate.

Next to embed the query, taken from input:
```py
query = input("Enter a question: ")
query_embedding = embed_chunks([query])
```

And then use it to query collection in weaviate:
```py
results = collection.query.near_vector(near_vector=query_embedding[0].tolist(), limit=5, return_properties=["text"])
```

## Day 3 - 2026-03-06


## Day 4 - 2026-03-09

Proper testing. Setup a sheet to start recording testing results properly:
Columns:

- Chunking Strategy
- Top k
- Query
- Chunks returned
- Chunks relevant
- System and User Prompt
- Answer
- Answer quality
- Input tokens
- Output tokens
- Notes

This was super useful. I straight away found a couple of bugs in the parsing logic. Highlighting once again, getting that right
across all edge cases is tricky and requires time, and testing.

The results were mixed - they can be seen in the accompanying sheet. Question is though how to quantify performance at this stage. 

There are standard ways of measuring a RAG systems performance. They are:

Recall, or Recall @ K: It measures how many of the total relevant documents in the knowledge base were actually returned by the retriever
Recall@K specifically looks at whether relevant documents appear within the first k results (e.g., Recall@10)

Precision: This measures how many of the returned documents are actually relevant.

Mean Average Precision (MAP): This metric evaluates the average precision for relevant documents within the first k results. It is used to assess ranking effectiveness and rewards the system for ranking relevant documents highly on the list.

Mean Reciprocal Rank (MRR): This measures how well the model performs at the very top of the ranking. It calculates the reciprocal of the rank of the first relevant document (e.g., if the first relevant doc is at rank 2, the score is 0.5) and averages this across many prompts

Let's start with Recall. The issue I have is, we have 4000 chunks, how do I know which chunks are relevant without going through them all manually?

I realised that this is where I can lean on the weaviate docs site search system.
All the results for these queries return high quality answers. Also returned is the document(s)
used for these answers. I can also return the document the chunk was found in and compare that with
the known list obtained from the document website result.

## Day 5 - 2026-03-09

I added a few things to the system to aid testing.

I tided up the Benchmarks tab in the sheet so it now just has the Query, the Result, and the sources.
I now use this to compare against the results from my system.

The tests are:
1 - Of the sources returned by my system, how many are the same as the ones returned by docs.weaviate.io
2 - I created an LLM-as-a-judge which compares the result my system returned, compared to the result returned by docs.weaviate.io and scores it out of 5.

I found that given the system just looks for all md/mdx files - files from _includes were being included, and these were getting returned, probably due to small document bias. I am still unable to to get full 5/5 answer quality scores.






















