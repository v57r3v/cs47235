java c
Homework 3: Coding portion 
AMATH   301,   Winter   2025 
Due   Friday,   January   31,   2025,   11:59PM   in   Gradescope 
20   points 
1.    (6   points)   In   Naive   Gaussian   elimination,   small   roundoff   error   can   become   amplified   and   lead   to   non-
sensical   results   when   a   pivot   value   is   small,   even   if   it’s   not   identically   zero.
We   wish   to   solve   the   system:

where   ϵ is   a   small, nonzero   number.    Because   the   first   pivot   is   ϵ,   Naive   Gaussian   elimination   will   require division   by   a   small   number   which   may   lead   to   large   rounding   error.Copy the   Naive   Gaussian elimination   and   full   Gaussian elimination   (with partial   pivoting)   codes   from   Canvas,      contained   within   gausselim   .ipynb.       Solve      the      system   Ax      =    b      using   the   Naive      Gaussian elimination   function,   and   we   will   call   the   solution   xnaive      (the   autograder   will   not   check   xnaive   ).   Then   solve Ax = b again   using   the   full   Gaussian   elimination   function,   which   we   call   xfull      (again,   not   checked   by   autograder).   If we   consider   xfull    to   be   the   true   value,   then   a   measure   of the   error   of xnaive      is:
error   =   ||xnaive   — xfull   ||   2
where

is   the   2-norm   of   x   where
x =   [   x1               x2               x3               x4               x5       ]T   .We will investigate what happens   as   ϵ shrinks.    Create   a np.array   object   called   epsvec   containing   the   values   1,   0.1,   0.01, ···   , 10 −15      (note:   decreasing   order).   For   each   such   value   of   ϵ,   determine   the   error   of   xnaive    as explained   above.    Store the   corresponding error values   in   a   np.array   object   called   errorvec.   Finally,   include   the   following   lines   of code   to   the   end   of your   code,   which   will   create   a   log-log   plot   of   log10   (ϵ)   on   the   x-axis   and   log10   (error)   on   the   y-axis.   The   general   trend   should   make   sense.
logeps      =      np.log10(epsvec)
logerror      =      np.log10(errorvec)
plt.plot(logeps,logerror,’-or’)
plt.xlabel(’$\log_{10}(\epsilon)$’)
plt.ylabel(’$\log_{10}(   ||x_{naive}-x_{full}||   _2)$’)
2.    (7 points) Begin with full Gaussian   elimination   code   from   gausselim   .ipynb   in   Canvas.   Alter   the   code   to   keep   track   of the   number   of row   flips   performed   in   the   process   of the   forward-elimination   steps.    If   we   want   to   look   for   trends   in   the   number   of   row   flips,   we   could   create   a   random   A   matrix   of   size   n   ×   n,   and   a   random   b   vector   of   size   n   ×   1,   perhaps   full   of   integers   from   0   to   99,   and   solve   Ax   =   b   using   Gaussian   elimination.
Use   the   following   code   to   create   such   random   A   and   b   matrices:
np   .random   .seed(1)      #ensures      your      random      numbers    are    the    same    as    the    autograder’s
A=np.random.randint(0,100,[n,n])
A=A.astype(np.float32)      #convert      integers      to      floating-point      objects    (decimals)
b=np.random.randint(0,100,[n,1])
if      n==256:
bseedcheck      =    bCreate   anp.array   object   nvec   which   contains   the   values   of   n   = 4,   代 写AMATH 301, Winter 2025 Homework 3: Coding portionMatlab
代做程序编程语言8, 16,   32,   64, 128,   256   (7 values   total).   For   each   value   of   n,   run   your   altered   full   Gaussian   elimination   code   to   record   the   percentage      of   rows which   were   flipped,   and   store   that   value   in   the   np.array   object   percentflips.    So,   percentflips   should   contain   7   values,   where   the   ith   value   is   the   percentage   of   rows   flipped,   when   the   length   of   b   is the   ith   value   of n.    By   “percentage”   we   mean   a   number   between   0   and   1   inclusive, not between   0   and   100.   The   general   trend   should   make   sense.Make   sure   to   reset   the   random   number   seed   to   1   before   creating   each   A,   but   do   not   reset   the   seed   to   1   before   creating   each   b   (in   other   words,   do   not   modify   the   order   or   content   of   the   lines   above–   if you   do,   your   random   numbers   will   differ   from   the   autograder’s.)    In   order   to   check   that   the   random   numbers   your   code   is   producing   are   the   same   as   the   autograder’s,   the   autograder   will   check   the   vector bseedcheck, which   is   your   vector   for   b   when   n   =   256.    If   that’s   correct,   it   assumes   that   all   the   random A   and   b’s   you   created   contain   the   right   numbers.
3.    (7 points) In   this   problem   we   will   solve   Ax   =   b   approximately   using   Jacobi   iteration.    Use   the   following code   to   create   a   random   b   vector   of   size   10   ×   1   as   a   np.array   object   called   bjacobi,   and   a   random strictly diagonally dominant A matrix of   size 10   ×   10 as anp.array object called Ajacobi.   The solution   x   is   a   10   × 1   vector.
np   .random   .seed(1)      #ensures      your      random      numbers    are    the    same    as    the    autograder’s
Ajacobi      =      0.01*np.random.randint(-50,51,[10,10])+10*np.eye(10)
bjacobi      =      np.random.randint(-50,51,[10,1])
Start   at   a   guess   of
x0    =   [   0         0         0 ··· 0   ]T
and   use   Jacobi   iteration   to   repeatedly   update   xn   ;   call   this   new   vector   xn+1   .    The   step   distance   from   xn    to   xn+1    can   be   quantified   by the   2-norm:
step   =   ||xn+1   — xn   ||   2Stop   the   Jacobi   iteration   as   soon   as   the   value   of the   step   is   below   10 −5   .    Store   the   values   of the   steps   in   a   np.array   object   called   stepvec.    So,   the   first   entry   of   stepvec   should   be   ||x1   — x0 ||   2   ,   the   second entry   is   ||x2   — x1 ||   2   ,   and   so   on;   the   last   entry   should   be   the   only   value   smaller   than   10 −5   .You   can   use   as   a   starting point   Kutz’s   code below, which   implemented   Jacobi iteration   on   the   system:

#      Jacobi      iterations
x      =      np.array([0]);      y      =      np.array([0]);      z      =      np.array([0])
for      j      in      range(100):
x      =      np.append(      x,    (7+y[j]-z[j])/4      )
y      =      np.append(      y,    (21+4*x[j]+z[j])/8    )
z      =      np.append(      z,    (15+2*x[j]-y[j])/5      )
if      abs(x[j+1]-x[j])      < 1e-5:
break 
print(j)
print(x[j      +      1])
print(y[j      +      1])
print(z[j      +      1])







         
加QQ：99515681  WX：codinghelp  Email: 99515681@qq.com
