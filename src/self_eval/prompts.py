#!/usr/bin/env python3

# (label: content) template
def structured_template(prectx: str,
                        tgt_sent: str,
                        ending: str,
                        tgt_word: str,
                        proposed_sense: str):
    return f'''
        PRE-CONTEXT: {prectx}
        TARGET SENTENCE: {tgt_sent}
        ENDING: {ending}
        TARGET WORD: {tgt_word}
        PROPOSED SENSE: {proposed_sense}
    '''

# natural language template
def nl_template(p,ts,e,tw,ps):
    return f'''
    story: {p}{ts}{e}
    what is the likelihood (from 1-5) that {tw} is used in the sense of {ps}?
    '''

# ---------------------------------------------------------------------------|
# System Prompts

# Prompt One --- Baseline
def base_sys_prompt() -> str:
    return '''
    You are given:
    1. a story, consisting of a pre-context, target sentence, and ending
    2. a target word
    3. a proposed semantic sense for that target word
    please provide a score out of a five point
    scale for how likely the target word matches the proposed sense given the story.
    You must provide your answer using JSON Schema with the following structure:
    score (1-5), thought, reasoning.
    '''
