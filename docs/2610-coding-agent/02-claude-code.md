## Design principle

### Singel Agent

### Terminal first

### Multiple Models

## Three Layers

| Layer          | Component                               | Objective                                 |
| -------------- | --------------------------------------- | ----------------------------------------- |
| UI             | REPL, Prompt, State                     | Terminal, State, Streaming                |
| Orchestration  | Query, Model manager,<br />Agent loader | Message routing, Model,<br />Agent loader |
| Tool execution | Tools, Permission                       | IO, Command, Permission                   |

### Multiple models cowork

- main, conversation
- task, task execution
- reasoning, thinking
- quick, simple NLP task

### Agent loader

- Tier 1: built-in agent
- Tier 2: ~/.claude/agents
- Tier 3: ./.claude/agents

### access file

read request -> file read tool , read and record timestamp , send file:read event

edit requst -> edit tool validation -> compare  current time vs record time; reject if current timer news -> write -> update timestamp -> send file:edit event

## query lifecycle

### instruction process

user input -> parse and validation -> build prompt

### AI reasoning

call LLM -> streaming response -> [tool call N] -> generate response

### tool exection

[toll call Y] Permission -> Tool execution -> recursive query (to call LLM )

## SubAgent

- task 1, code explorer subagnet
- task 2, arch subagent
- task 3, code review subagent

## Memory and Context

| type                        | lifecycle     | frequency    | strategy   |
| --------------------------- | ------------- | ------------ | ---------- |
| current conversation        | in session    | high         | memory     |
| conversation history sumary | in session    | middle       | compressed |
| project perference          | cross session | each session | file       |
| domain knowledge            | forever       | as required  | skills     |

### memory level

short, memory (message)

middle, compressed

long-term, claude.md

skill, skills

## plugins

### what:customzied workflow

### offical plugins

| name              | desc              | components                            |
| ----------------- | ----------------- | ------------------------------------- |
| feature-dev       | 7 steps dev       | command + 3 agents                    |
| code-review       | PR code review    | command + 5 concurrent agents         |
| security-guidance | security reminder | PreToolUse hook                       |
| hookify           | visualize hook    | commandd + conversation analyze agent |
| plugin-dev        | puglin dev        | command + 7 skills                    |
| commit-commands   | Git workflow      | git commands                          |

### sample: feature-dev

<span style="color:red">requirement -> explorer code repository -> classify questions -> arch design -> implementation -> quality review -> retrospective</span>

<span style="color:red">code-explorer agent: entrance, trace code reference and data flow, identify design pattern and arch, read 5~10 necessary files</span>

<span style="color:red">code-arch agent: analyze code pattern, design functionality, propose design sepc, concurrent solution</span>

<span style="color:red">code-revew agent: confidence score, check project spec, identify bug and vulnerable, code quality analyze</span>

## prompt engineering

### xml tag

```XML
<system-reminder>...</system-reminder>
<example>...</example>
<env>...</env>
```

### markdown

```Markdown
# task management

# tone and style

# following conventions
```

### layered prompts

layer 1: 身份定位，安全边界；

layer 2: 

Concise

```Markdown
you should be concise, and to the point
you MUST answer consisely with fewer than 4 lines (not including tool use or code generation), unless user ask for details
IMPORTANT: you should minimize output tokens as much as possible while maintaining helpfulness, quality, and accuracy
```

No amble

```Markdown
IMPORTANT: You should NOT answer with uncessary premable or postamble 
(such as explaining your code or summarizing your action), unless the user asks you to.

You Must avoid text before/after your response, such as "The answer is <answer>"

"Here is the content of the file ... " or "Based on the infomation provided, the answer is ... "
```

few-shot learning

```XML
<example>
  user: 2+2
  assistant: 4
</example>
<example>
  user: what command should i run to list files in the current directory?
  assistant: ls
</example>
```

No preachy

```Markdown
if you cannot or will not help the user with something, please do not say why or what it could lead to, since this comes across as prechy and annoying

please offer helpful alternatives if possible, and otherwise keep your reponse to 1-2 sentences
```

Transparency

```Markdown
when you run a non-trieval bash command, you shold explain what the comand does and why you are running it, 
to make sure  the user understands what you are doing (this is especially important when you are running 
a command that will make chagnes to the user's sysetm)
```

layer 3

- boundry
- taks management
- standard workflow

layer 4

- code convention
- code comment
