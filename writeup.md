# Reverse-Engineering Contextual Analogy-Making

## To Do

- Implement batch activation patching
    - Will need to incorporate padding to get tokens to line up properly.
- Implement path patching

## Problem

A common use-case for LLMs is loading a lot of text into context and expecting the model to use creative reasoning to generate insightful information. Here, we define creative reasoning as any process which produces novel and useful information (Stein, 1953). Analogy-making, the process of identifying and mapping structural similarities between different domains that supports further inference (Gentner, 1983), is a common form of creative reasoning. Behaviorally, we know LLMs are capable of analogy-making. But we lack a mechanistic understanding of how they do it.

## Hypotheses

- There is some set of attention heads responsible for doing the relational mapping.
- There is a representation in the final S1 and S2 tokens which contains the domains D1 and D2, which contain the objects, predicates, and truth value of the statements.
    - Possible Outcome 1: I'm able to probe for a representation of everything I expect at those tokens.
    - Possible Outcome 2: I don't find anything obvious and I have to try something else.

## High-Level Takeaways

- Most interesting parts of the project

## Key Experiments

- Behavioral success rate
- These heads perform the task
- This is what their attention pattern looks like
- This is how performance changes when we intervene on the heads
    - Ablate them
    - Isolate them

## Needs

- A bunch of graphs.
    - Bar charts of behavioral analysis
    - Heatmaps of activation patching
        - Residual, mlp, attention patching
        - Attention head patching
        - Path patching
- Enough detail to follow what I did without reading my code.
    - 1 - 3 pages.
- 

### Exploration

#### Prompt Testing

#### Behavioral Analysis

#### Attention Patterns (Exploratory)

- Just looking at the attention pattern on the prompt at random layers
    - Layer 0:
        - The final "to" token is looking at the subject and object tokens from the analogy, as I would expect. Most saliently the last one just before itself.
        - The first subject/object token of the final analogy is looking at the "is" and "of" of the predicates.
- So now I have a way to visualize the attention from any token to all the tokens preceding it at any given layer and head.
- So what I want to do is I want to find the most important layers and heads to do this for. Eventually I will path patch and try to find a circuit most responsible for completing this task. But for now I just want to do some activation patching and try to localize the behavior that way.

### Understanding

#### Activation Patching

- This will allow me to find the layer(s) most useful for more exploratory attention pattern observation.

#### Path Patching

- This will allow me to do very precise attention pattern observation.

### Distillation
- Writeup