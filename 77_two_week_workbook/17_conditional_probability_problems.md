# Conditional probability: worked problems

These are the problems that get asked out loud, at a whiteboard, with no paper table to lean on. The interviewer wants to see the setup more than the number, because the number follows once the setup is right. The single most common failure is writing the wrong conditional: you compute the probability of the evidence given the hypothesis when you were asked for the hypothesis given the evidence, or you condition on less than the problem told you. Therefore build one habit. Name the events in symbols first, write what is given and what is asked, and only then touch arithmetic. Every problem below is worked that way, and every number is checked by simulation.

## The three tools

**Bayes' rule**

$$P(H \mid E) = \frac{P(E \mid H)\,P(H)}{P(E)}$$

Use it whenever the question reverses the direction of a conditional you were handed.

**Law of total probability**

$$P(E) = \sum_{i} P(E \mid H_i)\,P(H_i) \quad \text{for a partition } \{H_i\}$$

Use it to build the denominator of Bayes' rule, or whenever the world splits into cases you know the rates for.

**Independence**

$$P(A \cap B) = P(A)\,P(B) \iff P(A \mid B) = P(A)$$

Use it only when the problem grants it; assuming it is the fastest way to a wrong answer.

## The problems

### P1. A disease affects 1 in 1000 people. A test has 99 percent sensitivity and 95 percent specificity. Your patient tests positive. What is the probability they have the disease?

**Set up.** Let $D$ be the event the patient has the disease and $+$ the event the test is positive. Given: $P(D) = 0.001$, $P(+ \mid D) = 0.99$ (sensitivity), $P(- \mid D^c) = 0.95$ (specificity), so $P(+ \mid D^c) = 0.05$. Asked for $P(D \mid +)$, which is the reverse of what was given.

**Work it.** Take 100,000 people. 100 are sick and 99 of them test positive. 99,900 are healthy and 5 percent of them, 4995 people, test positive anyway. So

$$P(D \mid +) = \frac{0.99 \times 0.001}{0.99 \times 0.001 + 0.05 \times 0.999} = \frac{0.00099}{0.05094}$$

**Answer.** analytic 0.0194, simulated 0.0194.

**Why it surprises people.** The false positives outnumber the true positives 50 to 1, because the healthy group is 999 times larger than the sick group and 5 percent of a huge number beats 99 percent of a tiny one.

> **Say it.** About two percent. I write the events down first: D for disease, plus for a positive test. I am given the test given the disease and asked for the disease given the test, so this is Bayes. Out of a hundred thousand people, a hundred are sick and ninety-nine of them test positive. Ninety-nine thousand nine hundred are healthy and five percent of those, nearly five thousand, also test positive. So ninety-nine true positives against about five thousand false ones, which is one in fifty. The base rate dominates.

### P2. Same test, run twice on the same patient, both positive. Now what?

**Set up.** Let $+_1$ and $+_2$ be the two positive results. Assume conditional independence given disease status: $P(+_1, +_2 \mid D) = P(+_1 \mid D) P(+_2 \mid D)$, and the same under $D^c$. Asked for $P(D \mid +_1, +_2)$.

**Work it.** The likelihoods square, the prior does not.

$$P(D \mid +_1, +_2) = \frac{0.99^2 \times 0.001}{0.99^2 \times 0.001 + 0.05^2 \times 0.999} = \frac{0.00098}{0.00348}$$

Equivalently, run Bayes twice: the posterior 0.0194 from P1 becomes the prior for the second test.

**Answer.** analytic 0.2818, simulated 0.2818.

**Why it surprises people.** Conditional independence is usually false here. If the patient has a condition that cross-reacts with the assay, both tests fail together, and the second test adds almost nothing. Repeating a test only helps when the errors are independent given the true state, which is exactly the assumption a good interviewer will ask you to defend.

> **Say it.** About twenty-eight percent, if I assume the two tests are conditionally independent given disease status. Under that assumption the likelihood ratio squares while the prior stays put, so the odds move from about one in fifty to about one in two point five. The clean way to say it is that the first posterior becomes the second prior. But I would flag the assumption: if the patient has something that cross-reacts with the assay, the two errors are correlated and the second test buys me far less than the arithmetic claims.

### P3. Two coins in a bag: one fair, one with heads on both sides. You draw one at random and flip it three times. All three come up heads. What is the probability you drew the double-headed coin?

**Set up.** Let $C_2$ be the event you drew the double-headed coin and $H_3$ the event of three heads. Given: $P(C_2) = P(C_2^c) = 0.5$, $P(H_3 \mid C_2) = 1$, $P(H_3 \mid C_2^c) = (1/2)^3 = 1/8$. Asked for $P(C_2 \mid H_3)$.

**Work it.**

$$P(C_2 \mid H_3) = \frac{1 \times 0.5}{1 \times 0.5 + 0.125 \times 0.5} = \frac{1}{1.125} = \frac{8}{9}$$

The prior cancels because it is symmetric, so the answer is just the likelihood ratio $8:1$ normalised.

**Answer.** analytic 0.8889, simulated 0.8890.

> **Say it.** Eight ninths. The prior is even, so the posterior odds equal the likelihood ratio. The double-headed coin produces three heads with probability one; the fair coin produces them with probability one eighth. So the odds are eight to one for the trick coin, which is eight ninths. Note it never reaches one, no matter how many heads I see — the fair coin can always get lucky. Each extra head just doubles the odds.

### P4. A family has two children. Version A: at least one is a boy. Version B: the older is a boy. In each case, what is the probability both are boys?

**Set up.** The sample space is ordered pairs (older, younger), each of $BB, BG, GB, GG$ with probability $1/4$. Version A conditions on the event $\{BB, BG, GB\}$. Version B conditions on the event $\{BB, BG\}$. The two conditioning events are different sets, so the two answers must differ.

**Work it.** Version A:

$$P(BB \mid \text{at least one boy}) = \frac{1/4}{3/4} = \frac{1}{3}$$

Version B:

$$P(BB \mid \text{older is a boy}) = \frac{1/4}{2/4} = \frac{1}{2}$$

**Answer.** Version A: analytic 0.3333, simulated 0.3325. Version B: analytic 0.5000, simulated 0.4992.

**Why it surprises people.** "At least one is a boy" is a statement about the pair, not about a named child. It removes only $GG$, leaving three equally likely cases. "The older is a boy" names a child and removes two cases, which leaves the other child's sex untouched at one half.

> **Say it.** One third and one half. I write the four ordered outcomes: boy-boy, boy-girl, girl-boy, girl-girl, each a quarter. "At least one is a boy" deletes only girl-girl, so three cases survive and one of them is both boys — one third. "The older is a boy" deletes both girl-first cases, leaving two, so it is one half. The difference is that the second version identifies a specific child, which makes the other child's sex independent of what I learned. The first version is a statement about the pair.

### P5. Same family, but now: at least one is a boy born on a Tuesday. Probability both are boys?

**Set up.** Extend the outcome to (sex, weekday) per child, so 14 equally likely values each, 196 ordered pairs. Let $E$ be the event that at least one child is a boy born on a Tuesday. Asked for $P(BB \mid E)$.

**Work it.** Count pairs containing at least one boy-Tuesday. By inclusion-exclusion over the two children, $14 + 14 - 1 = 27$ of the 196 pairs qualify. Of those, restrict the other child to be a boy: the other child has 7 boy-weekdays, so $7 + 7 - 1 = 13$ pairs are boy-boy.

$$P(BB \mid E) = \frac{13/196}{27/196} = \frac{13}{27}$$

**Answer.** analytic 0.4815, simulated 0.4803.

**Why it surprises people.** An apparently irrelevant fact, the weekday, moves the answer from $1/3$ toward $1/2$. It does so because it makes the conditioning event more specific, which makes the "two different children could satisfy it" double-count smaller relative to the total.

> **Say it.** Thirteen twenty-sevenths, just under one half. I extend each child to a sex-weekday pair, fourteen options, so a hundred ninety-six ordered pairs. At least one boy born Tuesday covers fourteen plus fourteen minus one, twenty-seven pairs, by inclusion-exclusion. Of those, thirteen have the other child also a boy. The weekday looks irrelevant but it sharpens the conditioning event, which shrinks the overlap term, and the answer slides from a third toward a half. That is the point of the problem: condition on exactly what you were told.

### P6. Monty Hall. You pick door 1. The host, who knows where the car is, opens door 3 to show a goat and offers you door 2. Do you switch?

**Set up.** Let $C_i$ be the event the car is behind door $i$, so $P(C_i) = 1/3$. Let $O_3$ be the event the host opens door 3. The host's rule matters: he never opens your door and never opens the car. So $P(O_3 \mid C_1) = 1/2$ (he picks between 2 and 3), $P(O_3 \mid C_2) = 1$ (forced), $P(O_3 \mid C_3) = 0$.

**Work it.**

$$P(C_1 \mid O_3) = \frac{(1/2)(1/3)}{(1/2)(1/3) + 1 \cdot (1/3) + 0} = \frac{1/6}{1/2} = \frac{1}{3}$$

so $P(C_2 \mid O_3) = 2/3$. Switch.

**Answer.** analytic 0.6667, simulated 0.6667 for switching.

**Why it surprises people.** People condition on "a goat was revealed", which is certain and therefore uninformative. The real evidence is which door the host was forced to open, and that is informative because his choice is constrained. If instead he opened a door at random and it happened to show a goat, $P(O_3 \mid C_2)$ drops to $1/2$ and the answer becomes even.

> **Say it.** Switch, two thirds. I condition on the host's actual action, not on the fact that a goat appeared. If the car is behind my door he has a free choice, so he opens door three half the time. If the car is behind door two he is forced to open door three. That factor of two is the whole evidence. So my door stays at a third and door two carries two thirds. If the host were choosing at random and just happened to miss the car, both likelihoods would be a half and it would be fifty-fifty.

### P7. Three prisoners, A, B and C. One will be pardoned, uniformly at random. A asks the guard to name one of the other two who will be executed. The guard names B. Should A feel better?

**Set up.** Let $P_A, P_B, P_C$ be the pardon events, each with probability $1/3$. Let $G_B$ be the event the guard names B. The guard never names A and never names the pardoned prisoner, and breaks ties fairly. So $P(G_B \mid P_A) = 1/2$, $P(G_B \mid P_B) = 0$, $P(G_B \mid P_C) = 1$.

**Work it.**

$$P(P_A \mid G_B) = \frac{(1/2)(1/3)}{(1/2)(1/3) + 0 + 1 \cdot (1/3)} = \frac{1}{3}, \qquad P(P_C \mid G_B) = \frac{2}{3}$$

**Answer.** analytic 0.3333 for A and 0.6667 for C; simulated 0.6668 for C.

**Why it surprises people.** A learns nothing about himself, because the guard was always going to name someone. All the information transfers to C. It is Monty Hall with the doors renamed, which is why recognising the structure matters more than remembering the answer.

> **Say it.** A should not feel better. His probability stays at a third. The guard was always going to name one of the other two, so the naming itself is certain and carries no news about A. But the likelihoods differ for C: if C is pardoned the guard must say B, while if A is pardoned the guard says B only half the time. So C goes from a third to two thirds. This is Monty Hall with different labels — the pardoned prisoner is the car and the guard is the host.

### P8. A factory has three machines. A makes 50 percent of output with a 1 percent defect rate, B makes 30 percent with 2 percent, C makes 20 percent with 3 percent. You pick a defective item. What is the probability it came from B?

**Set up.** Let $A, B, C$ partition the items by machine and let $F$ be the event the item is defective. Given the shares $P(A) = 0.5$, $P(B) = 0.3$, $P(C) = 0.2$ and the rates $P(F \mid A) = 0.01$, $P(F \mid B) = 0.02$, $P(F \mid C) = 0.03$. Asked for $P(B \mid F)$.

**Work it.** Law of total probability for the denominator:

$$P(F) = 0.5(0.01) + 0.3(0.02) + 0.2(0.03) = 0.005 + 0.006 + 0.006 = 0.017$$

$$P(B \mid F) = \frac{0.006}{0.017} = \frac{6}{17}$$

**Answer.** analytic 0.3529, simulated 0.3529.

**Why it surprises people.** C has the worst defect rate but contributes the same defect mass as B, because volume multiplies rate. The posterior ranks by the product, not by the rate.

> **Say it.** Six seventeenths, about thirty-five percent. I use the law of total probability to build the denominator: half times one percent is five in a thousand, point three times two percent is six, point two times three percent is six. Total seventeen defective per thousand items. B contributes six of those, so six over seventeen. The thing to notice is that C has the worst rate but produces exactly as many defects as B, because volume multiplies rate. The posterior over sources is proportional to share times defect rate.

### P9. A spam filter. Forty percent of mail is spam. The word "free" appears in 20 percent of spam and 2 percent of legitimate mail. An email contains "free". Probability it is spam?

**Set up.** Let $S$ be spam and $W$ the event the word appears. Given $P(S) = 0.4$, $P(W \mid S) = 0.2$, $P(W \mid S^c) = 0.02$. Asked for $P(S \mid W)$.

**Work it.**

$$P(S \mid W) = \frac{0.2 \times 0.4}{0.2 \times 0.4 + 0.02 \times 0.6} = \frac{0.08}{0.092} = \frac{20}{23}$$

In odds form: prior odds $0.4/0.6 = 2/3$, likelihood ratio $0.2/0.02 = 10$, posterior odds $20/3$, which is $20/23$.

**Answer.** analytic 0.8696, simulated 0.8686.

> **Say it.** About eighty-seven percent. I find odds easier here. The prior odds of spam are two to three. The likelihood ratio for the word is twenty percent over two percent, so ten. Multiply and I get posterior odds of twenty to three, which is twenty out of twenty-three. Naive Bayes is exactly this with one likelihood ratio per word, multiplied together — and that product is where the false independence assumption lives, because words in real mail are strongly correlated.

### P10. Your team runs A/B tests at 5 percent significance with 80 percent power. Historically about 1 in 10 tested ideas actually works. A test comes back significant. What is the probability the effect is real?

**Set up.** Let $R$ be the event the effect is real and $G$ the event the test is significant. Given $P(R) = 0.10$, $P(G \mid R) = 0.80$ (power), $P(G \mid R^c) = 0.05$ (the significance level). Asked for $P(R \mid G)$, which is not the p-value and not one minus the p-value.

**Work it.**

$$P(R \mid G) = \frac{0.8 \times 0.1}{0.8 \times 0.1 + 0.05 \times 0.9} = \frac{0.08}{0.125}$$

**Answer.** analytic 0.6400, simulated 0.6404.

**Why it surprises people.** A 5 percent significance level does not mean a 5 percent chance of being wrong. Here 36 percent of significant results are false. Lower the hit rate to 1 in 100 and the false discovery rate rises above 85 percent.

> **Say it.** Sixty-four percent, so more than a third of the significant results are false. The significance level is the probability of a positive given no effect, which is the wrong direction. I want the probability of an effect given a positive, so I need the prior. One in ten ideas work: point one times point eight power gives point zero eight true positives, point nine times point zero five gives point zero four five false ones. That is a thirty-six percent false discovery rate. It gets worse fast as the hit rate drops.

### P11. An urn has 4 red and 6 blue balls. You draw two without replacement. What is the probability the second is red? And what if the first was red?

**Set up.** Let $R_1$ and $R_2$ be the events that the first and second draws are red. Asked for $P(R_2)$ unconditionally, and $P(R_2 \mid R_1)$.

**Work it.** By the law of total probability,

$$P(R_2) = P(R_2 \mid R_1)P(R_1) + P(R_2 \mid R_1^c)P(R_1^c) = \frac{3}{9}\cdot\frac{4}{10} + \frac{4}{9}\cdot\frac{6}{10} = \frac{12 + 24}{90} = \frac{4}{10}$$

Conditionally, after a red is removed 3 of the remaining 9 are red, so $P(R_2 \mid R_1) = 1/3$.

**Answer.** $P(R_2)$: analytic 0.4000, simulated 0.3997. $P(R_2 \mid R_1)$: analytic 0.3333, simulated 0.3334.

**Why it surprises people.** The unconditional answer equals the first-draw probability by symmetry: before you look at anything, every ball is equally likely to be in position two. The draws are exchangeable but not independent, and people conflate the two.

> **Say it.** Four tenths unconditionally, one third given the first was red. The unconditional answer follows from symmetry — the second position is just as likely to hold any given ball as the first, so it is the same four in ten. I can also get it by the law of total probability and the fractions collapse back to four tenths. Given a red already gone, three of nine remain red, so one third. The lesson is that exchangeable is not the same as independent.

### P12. How many people do you need for a better-than-even chance that two share a birthday? Derive it, do not recall it.

**Set up.** Let $A_k$ be the event that the first $k$ people all have distinct birthdays, with 365 equally likely days and independent birthdays. Asked for the smallest $k$ with $P(A_k^c) > 0.5$.

**Work it.** Chain the conditionals. Person 1 is free. Given the first $k-1$ are distinct, person $k$ must avoid $k-1$ occupied days:

$$P(A_k) = \prod_{i=1}^{k-1} P(\text{person } i{+}1 \text{ distinct} \mid A_i) = \prod_{i=1}^{k-1}\frac{365-i}{365}$$

At $k = 23$ this product is 0.4927, so the collision probability is 0.5073.

**Answer.** analytic 0.5073 at 23 people, simulated 0.5072.

**Why it surprises people.** People count people and the answer scales with pairs. Twenty-three people give 253 pairs, each colliding with probability $1/365$, so the expected number of collisions is about 0.69 — order one. The general rule is a collision at roughly $\sqrt{d}$ draws from $d$ values.

> **Say it.** Twenty-three. I build it as a chain of conditionals on the complement. The first person is free, the second must avoid one day, the third must avoid two, and so on, so the probability of all distinct is the falling product of 365 minus i over 365. That drops below a half at twenty-three, giving a collision chance of about 0.507. It feels too small because collisions come from pairs, and twenty-three people make two hundred fifty-three pairs. The general scaling is root d, which is the birthday bound in hashing and cryptography.

### P13. You have 3 chips. Each round you win a chip with probability 0.6 and lose one with probability 0.4. You stop at 0 chips or at 5. What is the probability you reach 5?

**Set up.** Let $h_k$ be the probability of reaching 5 before 0, starting from $k$ chips. Boundary conditions $h_0 = 0$ and $h_5 = 1$. Condition on the first step: from $k$ you move to $k+1$ with probability $p = 0.6$ or to $k-1$ with probability $q = 0.4$, and then the problem restarts from the new position.

**Work it.** First-step conditioning gives the recursion

$$h_k = p\,h_{k+1} + q\,h_{k-1}$$

Its solution with $r = q/p = 2/3$ is

$$h_k = \frac{1 - r^k}{1 - r^N} = \frac{1 - (2/3)^3}{1 - (2/3)^5} = \frac{0.70370}{0.86831}$$

For a fair walk, $p = q$, the recursion is linear and $h_k = k/N$ instead.

**Answer.** analytic 0.8104, simulated 0.8103.

**Why it surprises people.** A 60 percent edge on each round turns a 3-versus-2 chip position into an 81 percent chance of finishing. Small per-step edges compound very hard over a bounded walk.

> **Say it.** About eighty-one percent. I condition on the first step: from k I go up with probability p and down with probability q, and then I face the same problem from the new state. That gives h k equals p times h k plus one plus q times h k minus one, with zero and one at the barriers. The solution is one minus r to the k over one minus r to the N, with r equal to q over p, here two thirds. Plugging in three and five gives 0.81. If the walk were fair, r is one and the answer collapses to k over N.

### P14. You roll a fair six-sided die repeatedly. What is the expected number of rolls until you have seen all six faces?

**Set up.** Let $T$ be the total number of rolls and split it as $T = \sum_{i=0}^{5} T_i$, where $T_i$ is the number of rolls made while you hold exactly $i$ distinct faces. Condition on the current count $i$: a roll is new with probability $(6-i)/6$, independently of everything before, so $T_i$ is geometric with that success probability.

**Work it.** A geometric waiting time with success probability $s$ has mean $1/s$, so $\mathbb{E}[T_i] = 6/(6-i)$. Linearity of expectation needs no independence between the stages:

$$\mathbb{E}[T] = \sum_{i=0}^{5}\frac{6}{6-i} = 6\left(1 + \tfrac12 + \tfrac13 + \tfrac14 + \tfrac15 + \tfrac16\right) = 6 H_6 = 6 \times 2.45$$

In general $\mathbb{E}[T] = n H_n \approx n(\ln n + \gamma)$.

**Answer.** analytic 14.7000, simulated 14.6999.

**Why it surprises people.** The last face alone costs 6 rolls on average, more than the first three stages combined. That is why the cost is $n \log n$ and not $n$.

> **Say it.** Fourteen point seven. I condition on how many distinct faces I already hold. With i faces in hand, each roll is new with probability six minus i over six, so that stage is geometric with mean six over six minus i. Sum the stage means by linearity — I do not need the stages to be independent — and I get six times the sixth harmonic number, which is six times 2.45. The last face alone costs six rolls on average, which is why the general answer is n log n rather than n.

### P15. A DNA profile from a crime scene matches the defendant. The expert says the chance of such a match in an unrelated person is 1 in a million. The prosecutor says there is therefore a one-in-a-million chance the defendant is innocent. What is wrong?

**Set up.** Let $G$ be guilt and $M$ the event of a profile match. The expert stated $P(M \mid G^c) = 10^{-6}$. The prosecutor asserted $P(G^c \mid M) = 10^{-6}$. Those are different conditionals. Getting from one to the other needs a prior. Suppose the offender is one of 1,000,000 adults in the city, so $P(G) = 10^{-6}$ before any evidence, and a guilty person always matches.

**Work it.**

$$P(G \mid M) = \frac{1 \times 10^{-6}}{1 \times 10^{-6} + 10^{-6} \times (1 - 10^{-6})} \approx \frac{1}{1 + 999999 \times 10^{-6}} = \frac{1}{1.999999}$$

About 999,999 innocent people are in the pool and roughly one of them matches by chance, so the match evidence alone leaves two plausible suspects.

**Answer.** analytic 0.5000, simulated 0.5003.

**Why it surprises people.** The rarity of the match is real, and it is still not the probability of innocence. The base rate of guilt is just as rare, and the two cancel. Any other evidence that narrows the suspect pool changes the answer enormously, which is exactly why the match figure must never be quoted alone.

> **Say it.** The prosecutor swapped the conditional. The expert gave the probability of a match given innocence; the prosecutor reported it as the probability of innocence given a match. Converting between them needs the prior odds of guilt. If the suspect pool is a million adults, the prior is one in a million, and about one innocent person in that pool matches by chance. So the match alone puts guilt at roughly fifty percent, not at 999,999 in a million. Narrow the pool with other evidence and the number moves a lot — which is the point.

### P16. Your classifier has 80 percent recall and 40 percent precision on a population with 5 percent positives. Fill in the confusion matrix for 10,000 items, then tell me the probability an item is truly negative given the model flagged it.

**Set up.** Let $Y$ be the true label and $\hat{Y}$ the prediction. Recall is $P(\hat{Y}{=}1 \mid Y{=}1) = 0.8$. Precision is $P(Y{=}1 \mid \hat{Y}{=}1) = 0.4$. Prevalence is $P(Y{=}1) = 0.05$. Asked for $P(Y{=}0 \mid \hat{Y}{=}1)$ and, as a follow-up, the negative predictive value $P(Y{=}0 \mid \hat{Y}{=}0)$.

**Work it.** In 10,000 items there are 500 positives. Recall gives $TP = 0.8 \times 500 = 400$, so $FN = 100$. Precision gives the total flagged: $TP / (TP + FP) = 0.4$, so $TP + FP = 1000$ and $FP = 600$. The remaining $9500 - 600 = 8900$ are true negatives.

$$P(Y{=}0 \mid \hat{Y}{=}1) = 1 - \text{precision} = \frac{600}{1000}, \qquad P(Y{=}0 \mid \hat{Y}{=}0) = \frac{8900}{8900 + 100}$$

**Answer.** flagged-but-negative: analytic 0.6000, simulated 0.5999. Negative predictive value: analytic 0.9889, simulated 0.9889.

**Why it surprises people.** The negative predictive value looks excellent, 98.9 percent, purely because negatives are 95 percent of the population. It would look good even for a useless model, so it is a bad headline metric at low prevalence.

> **Say it.** Sixty percent of the flags are wrong, because precision is forty percent and those are complements of each other. To build the matrix I start from ten thousand items: five hundred positives, recall eighty percent gives four hundred true positives and a hundred false negatives. Precision forty percent means the four hundred are forty percent of everything flagged, so a thousand flags total and six hundred false positives. That leaves eight thousand nine hundred true negatives. The negative predictive value is 98.9 percent, but that is mostly the base rate talking, not the model.

### P17. You roll a fair die. You may keep the value, or reroll once and must keep the second value. Play optimally. What is the expected payoff?

**Set up.** Let $X_1$ be the first roll and $X_2$ the second. Condition on the first roll. The reroll is worth $\mathbb{E}[X_2] = 3.5$, and that value does not depend on $X_1$, so the optimal rule is: keep $X_1$ if $X_1 > 3.5$, otherwise reroll. Let $V$ be the payoff under this policy.

**Work it.** By the law of total expectation, splitting on the first roll,

$$\mathbb{E}[V] = P(X_1 \ge 4)\,\mathbb{E}[X_1 \mid X_1 \ge 4] + P(X_1 \le 3) \times 3.5 = \tfrac12 \times 5 + \tfrac12 \times 3.5$$

**Answer.** analytic 4.2500, simulated 4.2484.

**Why it surprises people.** The threshold is the continuation value, not the median or the mean of the final payoff. With two rerolls allowed the threshold rises to 4.25, so you would then keep only a 5 or a 6.

> **Say it.** Four and a quarter. I condition on the first roll and compare it to the value of continuing, which is the plain mean of a die, three point five. So I keep a four, five or six and reroll a one, two or three. Half the time I average five, half the time I average three point five, so the answer is four point two five. The general principle is that the stopping threshold equals the continuation value, and with more rerolls allowed that threshold rises — with two rerolls it becomes 4.25, so only fives and sixes are kept.

## Simulating a conditional

The pattern is always the same: sample the whole world, filter the rows where the conditioning event holds, then take the mean of the target event on the survivors. Sample the cause first and the evidence conditioned on it, never the other way round. Applied to P1.

```python
import numpy as np

def conditional_mc(sample_fn, target_fn, given_fn, n=2_000_000, seed=0):
    """Estimate P(target | given) by sampling, filtering, then averaging."""
    rng = np.random.default_rng(seed)
    world = sample_fn(rng, n)                 # one full draw of the world per row
    keep = given_fn(world)                    # conditioning is a boolean filter
    return float(target_fn(world)[keep].mean()), int(keep.sum())

PREV, SENS, SPEC = 0.001, 0.99, 0.95   # problem P1's three given numbers

def sample_fn(rng, n):
    sick = rng.random(n) < PREV               # draw the cause first
    u = rng.random(n)                         # then the test, conditioned on the cause
    positive = np.where(sick, u < SENS, u < 1 - SPEC)
    return {"sick": sick, "positive": positive}

p_hat, kept = conditional_mc(sample_fn,
                             lambda w: w["sick"],
                             lambda w: w["positive"])
analytic = SENS * PREV / (SENS * PREV + (1 - SPEC) * (1 - PREV))
print("kept       ", kept)
print("simulated  ", round(p_hat, 5))
print("analytic   ", round(analytic, 5))
```

Ran with 2,000,000 samples: 101,760 rows survived the filter, giving `0.01957` against the closed form `0.01943`. The standard error of the estimate is $\sqrt{p(1-p)/m}$ over the $m$ surviving rows, so a rare conditioning event needs a very large $n$; for P1 that is about 0.0004, and the gap is well inside it. When the conditioning event is very rare, as in P15, simulate the counts directly with a binomial instead of materialising the rows.

## The traps, collected

Five errors account for almost every wrong answer. The first is transposing the conditional: reporting $P(E \mid H)$ when the question asked for $P(H \mid E)$. That is the prosecutor's fallacy in P15 and the p-value misreading in P10, and the fix is to write both symbols down before computing. The second is dropping the base rate, as in P1, where a 99 percent sensitive test still gives a 2 percent posterior because the healthy population is enormous. The third is conditioning on less than you were told: in Monty Hall the informative fact is which constrained door the host opened, not that a goat appeared, and in P5 an apparently irrelevant weekday genuinely moves the answer. The fourth is assuming independence the problem never granted, as with the repeated test in P2 or the words in a naive Bayes filter. The fifth is dropping the normalising denominator, which is the law of total probability summed over every case, and which is what makes the posterior a probability rather than a score.

## Done when

- You can state the events in symbols, and say which conditional is given and which is asked, for any of these 17 problems within 15 seconds of hearing it.
- You can work P1, P8 and P10 to a number in your head using a population table of 10,000 or 100,000, with no algebra.
- You can derive the Monty Hall posterior from the three host likelihoods, and say exactly what changes when the host chooses at random.
- You can write the sample-filter-mean Monte Carlo checker from memory and have it agree with your analytic answer on the first run.
