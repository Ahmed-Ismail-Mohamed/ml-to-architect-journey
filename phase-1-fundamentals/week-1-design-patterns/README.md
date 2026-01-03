# Week 1: Design Patterns & SOLID Principles

## 🎯 Week Overview

**Goal**: Transform your ML code from rigid to modular

### This Week You'll:
- ✅ Master SOLID principles with ML examples
- ✅ Implement Adapter pattern for swappable LLMs
- ✅ Use Strategy pattern for different retrieval methods
- ✅ Apply Factory pattern for component creation
- ✅ Refactor real ML code

---

## 📚 Daily Learning Plan

### **Day 1: SOLID Principles (Monday)**
- 📖 Read: [SOLID Principles for ML Engineers](./lessons/01-solid-principles.md)
- 🎥 Watch: Uncle Bob SOLID principles talk
- 💻 Code: Review bad vs good examples

**Time**: 2-3 hours

---

### **Day 2: Adapter Pattern (Tuesday)**
- 📖 Read: [Adapter Pattern - Swap LLM Providers](./lessons/02-adapter-pattern.md)
- 💻 Exercise: [Build Model Provider Adapter](./exercises/exercise-1-model-provider-adapter.md)
- 🎯 Goal: Swap OpenAI ↔ Claude in 2 lines of code

**Time**: 3-4 hours

---

### **Day 3: Strategy Pattern (Wednesday)**
- 📖 Read: [Strategy Pattern - Multiple Algorithms](./lessons/03-strategy-pattern.md)
- 💻 Exercise: [Implement Chunking Strategies](./exercises/exercise-2-chunking-strategies.md)
- 🎯 Goal: Switch chunking methods without code changes

**Time**: 3-4 hours

---

### **Day 4: Factory Pattern (Thursday)**
- 📖 Read: [Factory Pattern - Dynamic Creation](./lessons/04-factory-pattern.md)
- 💻 Exercise: [Build Component Factory](./exercises/exercise-3-component-factory.md)
- 🎯 Goal: Create components from configuration

**Time**: 3-4 hours

---

### **Day 5: Dependency Injection (Friday)**
- 📖 Read: [Dependency Injection for Testability](./lessons/05-dependency-injection.md)
- 💻 Exercise: [Refactor Your ML Code](./exercises/exercise-4-refactor-ml-code.md)
- 🎯 Goal: Make your code fully testable

**Time**: 4-5 hours

---

### **Weekend: Build Your Project**
- 🏗️ Project: [Multi-Provider LLM Wrapper](./project/README.md)
- 📝 Reflection: Write your weekly reflection
- ✅ Update progress tracker

**Time**: 4-6 hours

---

## 🎓 Key Concepts

### **SOLID Principles**
1. **S**ingle Responsibility - One class, one job
2. **O**pen/Closed - Open for extension, closed for modification
3. **L**iskov Substitution - Subtypes must be substitutable
4. **I**nterface Segregation - Many specific interfaces > one general
5. **D**ependency Inversion - Depend on abstractions, not concretions

### **Design Patterns**
- **Adapter**: Convert one interface to another
- **Strategy**: Encapsulate algorithms, make them interchangeable
- **Factory**: Create objects without specifying exact class

---

## 💻 Exercises

All exercises are in `./exercises/`:

1. **Model Provider Adapter** - Swap LLM providers
2. **Chunking Strategies** - Different text splitting methods
3. **Component Factory** - Dynamic component creation
4. **Refactor ML Code** - Apply all patterns to real code

---

## 🏆 Week 1 Challenge

**Refactor your existing churn prediction project**:
- Extract LLM logic behind an interface
- Make it swappable (OpenAI → Gemini → Claude)
- Add different prompt strategies
- Make it fully testable
- Write tests!

Compare before & after architecture.

---

## 📊 Success Criteria

By Friday, you should be able to:

- [ ] Explain each SOLID principle with ML examples
- [ ] Swap LLM providers in your code with 2 lines
- [ ] Switch retrieval strategies without touching core logic
- [ ] Create components from config files
- [ ] Write unit tests for your ML code
- [ ] Identify SOLID violations in existing code

---

## 🚀 Let's Begin!

👉 **Start here**: [Lesson 1 - SOLID Principles](./lessons/01-solid-principles.md)

---

## 💡 Pro Tips

1. **Relate to your experience**: Think about pain points in your ML projects
2. **Refactor, don't rewrite**: Apply patterns to existing code
3. **Test everything**: If you can't test it, it's not modular enough
4. **Ask "what if"**: What if I need to change this provider? Database? Algorithm?

**Happy coding!** 🎯