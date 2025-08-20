### 🧩 What is JSON?

**JSON** stands for  **JavaScript Object Notation** .

It’s a **text-based format** for storing and exchanging data.

* Think of it as a **universal way to write data** that both humans and machines can read.
* It’s commonly used in **APIs, web apps, configs, and databases** to send/receive structured data.

---

### Example of JSON

Here’s a small JSON file:

<pre class="overflow-visible!" data-start="442" data-end="667"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-json"><span><span>{</span><span>
  </span><span>"name"</span><span>:</span><span></span><span>"your-name"</span><span>,</span><span>
  </span><span>"age"</span><span>:</span><span></span><span>your-age</span><span>,</span><span>
  </span><span>"is_student"</span><span>:</span><span></span><span>true</span><span></span><span>,</span><span>
  </span><span>"languages"</span><span>:</span><span></span><span>[</span><span>"Persian"</span><span>,</span><span></span><span>"English"</span><span>,</span><span></span><span>"Python"</span><span>]</span><span>,</span><span>
  </span><span>"university"</span><span>:</span><span></span><span>{</span><span>
    </span><span>"name"</span><span>:</span><span></span><span>"Shiraz University of Technology"</span><span>,</span><span>
    </span><span>"major"</span><span>:</span><span></span><span>"Computer Engineering"</span><span>
  </span><span>}</span><span>
</span><span>}</span><span>
</span></span></code></div></div></pre>

---

### Rules of JSON

1. **Data is in key-value pairs** → `"key": value`
2. **Keys must be strings** (inside quotes).
3. **Values can be** :

* String → `"text"`
* Number → 22
* Boolean → `true` or `false`
* Null → `null`
* Array (list) → `["a", "b", "c"]`
* Object (dictionary) → `{"nested": "data"}`

1. Uses **curly braces `{}`** for objects and **square brackets `[]`** for lists.
2. It’s language-independent (works in Python, Java, C++, JavaScript, etc.).

---

### JSON vs Python Dictionary

They look very similar, but:

* **JSON** is text (like a `.txt` file).
* **Python dict** is an in-memory object.

Example:

<pre class="overflow-visible!" data-start="1309" data-end="1437"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-python"><span><span># Python dictionary</span><span>
data = {</span><span>"name"</span><span>: </span><span>"your-name"</span><span>, </span><span>"age"</span><span>: </span><span>22</span><span>}

</span><span># JSON string (text)</span><span>
</span><span>'{ "name": "your-name", "age": 22 }'</span><span>
</span></span></code></div></div></pre>

We use Python’s `json` module to  **convert between the two** .


### 🧩 Create a virtual environment

<pre class="overflow-visible!" data-start="168" data-end="201"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python -m venv venv
</span></span></code></div></div></pre>

* `python3` → calls Python (use `python` on Windows if needed).
* `-m venv` → uses Python’s built-in venv module.
* `venv` → the folder name for the environment (you can name it anything).

This creates a directory `venv/` with its own Python interpreter and packages.


* **On Windows (Command Prompt):**

<pre class="overflow-visible!" data-start="638" data-end="671"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-cmd"><span>venv\Scripts\activate</span></code></div></div></pre>


When you’re done, exit the environment with:

<pre class="overflow-visible!" data-start="992" data-end="1014"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>deactivate</span></span></code></div></div></pre>


# 🤖 Chatbot with Memory – Homework Guide

Welcome to your **Chatbot with Memory** project! The goal of this homework is to **build a simple Python chatbot** that remembers the conversation and can answer questions based on previous messages.

---

## 📝 Objectives

By the end of this homework, you should be able to:

1. Call the OpenAI API to generate chatbot responses.
2. Keep track of the conversation so the chatbot “remembers” previous messages.
3. Interact with the chatbot in a terminal session.
4. Save conversation history if you like (optional stretch goal).

## 💻 Running the Chatbot

Run the Python script:

<pre class="overflow-visible!" data-start="1584" data-end="1613"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre! language-bash"><span><span>python chatbot.py
</span></span></code></div></div></pre>

* You will see a prompt: `You:`
* Type anything to chat with the bot.
* The chatbot will remember your previous messages.
* To quit, type:

<pre class="overflow-visible!" data-start="1760" data-end="1772"><div class="contain-inline-size rounded-2xl relative bg-token-sidebar-surface-primary"><div class="sticky top-9"><div class="absolute end-0 bottom-0 flex h-9 items-center pe-2"><div class="bg-token-bg-elevated-secondary text-token-text-secondary flex items-center gap-4 rounded-sm px-2 font-sans text-xs"><span class="" data-state="closed"></span></div></div></div><div class="overflow-y-auto p-4" dir="ltr"><code class="whitespace-pre!"><span><span>quit</span><span>
</span></span></code></div></div></pre>

---

## 🧩 How It Works (High-Level)

* **conversation_history** → A Python list that stores messages (both yours and the bot’s).
* Each new user input is added to this list.
* The full conversation history is sent to the OpenAI API for context.
* The chatbot replies and its response is added to the history.

---

## 🎯 Stretch Goals

1. **Persistent memory**
   * Save `conversation_history` to a JSON file.
   * Reload it when restarting the bot so it “remembers” past sessions.
2. **Multiple users**
   * Extend the code to handle multiple conversation sessions.
3. **Custom personality**
   * Change the system message to give the chatbot a personality, e.g. “You are a friendly tutor for Python students.”

---

## ✅ Deliverables

* `chatbot.py` script that runs the chatbot.
* (Optional) `history.json` if you implement persistent memory.
* A short note explaining your stretch goal implementation (if done).
