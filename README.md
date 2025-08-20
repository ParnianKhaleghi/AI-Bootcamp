### What is JSON?

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
