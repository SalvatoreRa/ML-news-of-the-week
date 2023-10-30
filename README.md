# ML news of the week

A collection of the best ML news every week (research, news, resources)

[Here](https://github.com/SalvatoreRa/tutorial), you can find articles and tutorials about artificial intelligence

For each week you will find different sections:
* 

# ML news: Week 23-29 October

## Research
|Link|description|
|---|---|
|[Geographical erasure in language generation](https://www.amazon.science/publications/geographical-erasure-in-language-generation) | LLMs encode a vast amount of knowledge but it is not representative of all countries, Amazon shows how to mitigate this unbalance|
|[Entangled Preferences: The History and Risks of Reinforcement Learning and Human Feedback](https://arxiv.org/abs/2310.13595) | A deep dive in the history of  RLHF, potential issues and suggestions for new lines of research|
|[AgentTuning: Enabling Generalized Agent Abilities for LLMs](https://huggingface.co/papers/2310.12823) | Open-source models are inferior as AI agents when you need them as efficient controllers for complex tasks. This paper  highlights how to create efficient agent LLaMA-2  [models](https://huggingface.co/THUDM/agentlm-70b)|
|[The Foundation Model Transparency Index](https://hai.stanford.edu/news/introducing-foundation-model-transparency-index?utm_source=tldrai) | Stanford's new index rates the transparency of 10 foundation model companies and finds them lacking. The new index analyses 100 parameters, showing there is room for improvements|
|[BotChat: Evaluating LLMs' Capabilities of Having Multi-Turn Dialogues](https://arxiv.org/abs/2310.13650v1) | evaluation of the ability of large language models (LLMs) to engage in human-like multi-turn conversations. |
|[SALMONN: Towards Generic Hearing Abilities for Large Language Models](https://arxiv.org/abs/2310.13289v1) | SALMONN understands text and audio at the same time, can be used for speech recognition and speech translation. [official code](https://github.com/bytedance/salmonn)|
|[FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling](http://haonanqiu.com/projects/FreeNoise.html?utm_source=tldrai) | While you can generate easily an image with diffusion creating a video is much more complex (consistency), this work allows generations up to 512 frames long [paper](https://arxiv.org/abs/2310.15169), [code](https://github.com/arthur-qiu/LongerCrafter)|
|[PDFTriage: Question Answering over Long, Structured Documents](https://arxiv.org/abs/2309.08872) | Finding information from pdfs (web pages or other multi-page structured documents) is more difficult than for regular text. Therefore researchers at Adobe Research have developed a model that is able to consider both the text and the structure of the document|
|[VidChapters-7M: Video Chapters at Scale](https://antoyang.github.io/vidchapters.html) |Segmenting long videos into chapters enables users to quickly navigate to the information of their interest. Here the authors collected VidChapters-7M, a dataset of 817K user-chaptered videos including 7M chapters in total. |
|[RLMRec: Representation Learning with Large Language Models for Recommendation](https://arxiv.org/abs/2310.15950) | In this article the authors enhanced a recommendation system with an LLM, resulting in better recommendations. [code here](https://github.com/hkuds/rlmrec)|
|[CommonCanvas: An Open Diffusion Model Trained with Creative-Commons Images](https://arxiv.org/abs/2310.16825) | We assemble a dataset of Creative-Commons-licensed (CC) images, which we use to train a set of open diffusion models that are qualitatively competitive with Stable Diffusion 2 (SD2). [official code](https://github.com/mosaicml/diffusion)|
|[LLM-FP4: 4-Bit Floating-Point Quantized Transformers](https://arxiv.org/abs/2310.16836v1) | We propose LLM-FP4 for quantizing both weights and activations in large language models (LLMs) down to 4-bit floating-point values, in a post-training manner. Existing post-training quantization (PTQ) solutions are primarily integer-based and struggle with bit widths below 8 bits.[official code](https://github.com/nbasyl/llm-fp4)|
|[Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://arxiv.org/abs/2310.17157) |For a specific input, only a small fraction of attention heads and MLP neurons are needed, while the rest can be "silenced" without changing the output. Deja Vu to speed up inference for large language models. exploiting "contextual sparsity" (finding small subsets of model parameters that are sufficient to compute the same output for a given input.).  This is unlike prior pruning methods that permanently remove parameters.  [official code](https://github.com/FMInference/DejaVu/tree/master) |
|[ConvNets Match Vision Transformers at Scale](https://arxiv.org/abs/2310.16764) | Many researchers believe that ConvNets perform well on small or moderately sized datasets, but are not competitive with Vision Transformers when given access to datasets on the web-scale. The authors invested the same computer budget on a CNN to make a fair comparison with the vision transformers and they matched the performance|
|[Llemma: An Open Language Model For Mathematics](https://arxiv.org/abs/2310.10631) |a large language model for mathematics, the authors show how using a small model in continuous pretraining you can beat bigger models on Math and STEM. [deep dive](https://levelup.gitconnected.com/llemma-a-model-speaking-math-c8c07e1c001c) |
|[Zephyr: Direct Distillation of LM Alignment](https://arxiv.org/abs/2310.16944) | a 7B parameter model with competitive performance to ChatGPT on AlpacaEval|

## News
|Link|description|
|---|---|
|[New Nvidia AI agent, powered by GPT-4, can train robots](https://venturebeat.com/ai/new-nvidia-ai-agent-powered-by-gpt-4-can-train-robots/) | Eureka, a new AI agent (powered by GPT-4) can teach complex skills to robots|
|[‘Mind-blowing’ IBM chip speeds up AI](https://www.nature.com/articles/d41586-023-03267-0) | IBM has developed a brain-inspired computer chip that could supercharge artificial intelligence (AI) by working faster with much less power  |
|[“Math is hard” — if you are an LLM – and why that matters](https://garymarcus.substack.com/p/math-is-hard-if-you-are-an-llm-and) | LLM success on math is still limited, especially if you just rely on a LLM|
|[Apple Rumored to Follow ChatGPT With Generative AI Features on iPhone as Soon as iOS 18](https://www.macrumors.com/2023/10/19/apple-generative-ai-late-2024-jeff-pu/) |Apple plans to start implementing generative AI technology on the iPhone and iPad in late 2024 at the earliest according to analysts |
|[Reddit can survive without search](https://www.theverge.com/2023/10/20/23925504/reddit-deny-force-log-in-see-posts-ai-companies-deals) | Reddit and other companies may stop crawlers (and be not find anymore on google search) if they do not find an agreement in generative AI|
|[This new data poisoning tool lets artists fight back against generative AI](https://www.technologyreview.com/2023/10/23/1082189/data-poisoning-artists-fight-generative-ai) |A new tool lets artists add invisible changes to the pixels in their art before they upload it online so that if it’s scraped into an AI training set, it can cause the resulting model to break in chaotic and unpredictable ways.  |
|[AI risk must be treated as seriously as climate crisis, says Google DeepMind chief](https://www.theguardian.com/technology/2023/oct/24/ai-risk-climate-crisis-google-deepmind-chief-demis-hassabis-regulation) | Demis Hassabis calls for greater regulation to quell existential fears over tech with above-human levels of intelligence|
|[Claude accessibility is expanded to 95 countries](https://twitter.com/AnthropicAI/status/1714025126516432996) | |
|[IBM Presents NorthPole](https://research.ibm.com/blog/northpole-ibm-ai-chip) | a new chip much faster for AI and much more energy efficient|
|[Perplexity raises new funding at $500 million valuation](https://techstartups.com/2023/10/24/ai-search-startup-perplexitys-valuation-climbs-to-500-million-after-new-funding-round-led-by-ivp/) | Perplexity is developing an AI-powered search engine competing with the likes of OpenAI’s ChatGPT and Google’s Bard. According to recent reports, Perplexity has been generating annual recurring revenue of $3 million as of this month.|
|[AI rapidly diagnoses brain tumours during surgery](https://www.nature.com/articles/d41586-023-03072-9) |A machine-learning method to assess DNA can accurately classify brain tumours in real time. This rapid analysis might help surgeons to identify the tumour type when operating and to adjust their surgical strategy accordingly. |
|[AI executive order on October 30](https://www.engadget.com/the-white-house-will-reportedly-reveal-a-sweeping-ai-executive-order-on-october-30-200558649.html) |The Biden Administration is reportedly set to unveil a broad executive order on artificial intelligence next week. |
|[Lenovo and NVIDIA Announce Hybrid AI Solutions to Help Enterprises Quickly Adopt GenAI](https://nvidianews.nvidia.com/news/lenovo-nvidia-hybrid-ai) |New End-to-End Solutions Include Accelerated Systems, AI Software and Expert Services to Build and Deploy Domain-Specific AI Models with Ease |


## Resources
|Link|description|
|---|---|
|[caption-usampling](https://github.com/sayakpaul/caption-upsampling) | DALL-3 power is derived from better data quality, this library can allow you to upsample your dataset |
|[SolidGPT](https://github.com/AI-Citizen/SolidGPT) | Chat everything with your code repository, ask repository-level code questions, and discuss your requirements. AI Scan and learning your code repository, provide you code repository level answer|
|[GoLLIE 34B](https://huggingface.co/HiTZ/GoLLIE-34B) | zero-shot Information Extraction model for extracting information from unstructured data (CSV, JSON, and so on)|
|[Arithmo-Mistral-7B](https://huggingface.co/akjindal53244/Arithmo-Mistral-7B) | Mistral 7B fine-tuned on math|
|[GraphMaker](https://github.com/Graph-COM/GraphMaker) |a diffusion model capable of generating highly realisitc large attributed graphs. [original article](https://github.com/Graph-COM/GraphMaker) |
|[Meta’s Habitat 3.0 simulates real-world environments for intelligent AI robot training](https://siliconangle.com/2023/10/20/metas-habitat-3-0-simulates-real-world-environments-intelligent-ai-robot-training/) |Researchers from Meta Platforms Inc.’s Fundamental Artificial Intelligence Research team said today they’re releasing a more advanced version of the AI simulation environment Habitat, which is used to teach robots how to interact with the physical world. |
|[SAM-Med3D](https://github.com/uni-medical/sam-med3d) |the most comprehensive study to modify SAM for 3D medical images. Curated the most extensive volumetric medical dataset to date for training, boasting 131K 3D masks and 247 categories. [paper](https://arxiv.org/abs/2310.15161)|
|[deepsparse](https://github.com/neuralmagic/deepsparse) | DeepSparse is a CPU inference runtime that takes advantage of sparsity to accelerate neural network inference. |
|[ExecuTorch](https://pytorch.org/blog/pytorch-edge/) |PyTorch Edge: Enabling On-Device Inference Across Mobile and Edge Devices with ExecuTorch |
|[Spelltest: AI-to-AI Testing for LLM Based Applications](https://github.com/artas728/spelltest) | Today's AI-driven applications largely depend on Large Language Models (LLMs) like GPT-4 to deliver innovative solutions. However, ensuring that they provide relevant and accurate responses in every situation is a challenge. Spelltest addresses this by simulating LLM responses using synthetic user personas and an evaluation technique to evaluate these responses automatically(but still requires human supervision).|
|[polyfire-js](https://github.com/polyfire-ai/polyfire-js) |An all-in-one managed backend for AI apps. Build AI apps from the frontend, very fast |
|[ToRA: A Tool-Integrated Reasoning Agent](https://github.com/microsoft/ToRA) |ToRA is a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical reasoning problems by interacting with tools, e.g., computation libraries and symbolic solvers. ToRA series seamlessly integrate natural language reasoning with the utilization of external tools, thereby amalgamating the analytical prowess of language and the computational efficiency of external tools. |
|[Adala](https://github.com/HumanSignal/adala/) |Adala offers a robust framework for implementing agents specialized in data processing, with an emphasis on diverse data labeling tasks. |

## Perspectives
|Link|description|
|---|---|
|[Emotional labor and its consequences](https://seths.blog/2023/10/emotional-labor-and-its-consequences/) | Emotional labor is what differentiate us from AI |
|[The Techno-Optimist Manifesto](https://a16z.com/the-techno-optimist-manifesto) | A blog post that has ignited a strong debate in Silicon Valley about positive impact of technology|
|[Peak Data](https://eastwind.substack.com/p/peak-data) | a blog post discussing what will happen if the internet is filled only with AI-generated data, this will lead probably to collapse of AI model trained on these data|
|[Five Areas of AI Opportunity According to Snowflake’s Ahmad Khan](https://lsvp.com/five-areas-of-ai-opportunity-according-to-snowflakes-ahmad-khan/) |Lightspeed recently hosted the latest in its Generative AI series in Los Angeles, a fireside chat with Ahmad Khan, Head of AI/ML Strategy at Snowflake |
|[An AI revolution is brewing in medicine. What will it look like?](https://www.nature.com/articles/d41586-023-03302-0) |Emerging generalist models could overcome some limitations of first-generation machine-learning tools for clinical use. |
|[The Convergence of Data & Software Engineering in the Age of AI](https://tomtunguz.com/data-engineering/) | This convergence signals how far data teams have evolved into core engineering teams. Machine learning’s demand for data has accelerated this movement because AI needs data to function.|
|[Managing AI Risks in an Era of Rapid Progress](https://managing-ai-risks.com/managing_ai_risks.pdf) | Soem of the biggest names in the field (Hinton, Bengio and so on) discuss the potential threats of AI and how to manage them |
