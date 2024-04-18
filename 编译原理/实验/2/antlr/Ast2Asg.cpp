#include "Ast2Asg.hpp"
#include <unordered_map>

#define self (*this)

namespace asg {

// 符号表，保存当前作用域的所有声明
struct Ast2Asg::Symtbl : public std::unordered_map<std::string, Decl*>
{
  Ast2Asg& m;
  Symtbl* mPrev;

  Symtbl(Ast2Asg& m)
    : m(m)
    , mPrev(m.mSymtbl)
  {
    m.mSymtbl = this;
  }

  ~Symtbl() { m.mSymtbl = mPrev; }

  Decl* resolve(const std::string& name);
};

Decl*
Ast2Asg::Symtbl::resolve(const std::string& name)
{
  auto iter = find(name);
  if (iter != end())
    return iter->second;
  ASSERT(mPrev != nullptr); // 标识符未定义
  return mPrev->resolve(name);
}

TranslationUnit*
Ast2Asg::operator()(ast::TranslationUnitContext* ctx)
{
  auto ret = make<asg::TranslationUnit>();
  if (ctx == nullptr)
    return ret;

  Symtbl localDecls(self);

  for (auto&& i : ctx->externalDeclaration()) {
    if (auto p = i->declaration()) {
      auto decls = self(p);
      ret->decls.insert(ret->decls.end(),
                        std::make_move_iterator(decls.begin()),
                        std::make_move_iterator(decls.end()));
    }

    else if (auto p = i->functionDefinition()) {
      auto funcDecl = self(p);
      ret->decls.push_back(funcDecl);

      // 添加到声明表
      localDecls[funcDecl->name] = funcDecl;
    }

    else
      ABORT();
  }

  return ret;
}

//==============================================================================
// 类型
//==============================================================================

Ast2Asg::SpecQual
Ast2Asg::operator()(ast::DeclarationSpecifiersContext* ctx)
{
  SpecQual ret = { Type::Spec::kINVALID, Type::Qual() };  // Type::Spec::kINVALID表示类型规范尚未设置，Type::Qual()创建了一个空的类型修饰符。

  for (auto&& i : ctx->declarationSpecifier()) {
    if(auto p = i->Const()) {
      ret.second.const_ = true;
    }
    if (auto p = i->typeSpecifier()) {
      if (ret.first == Type::Spec::kINVALID) {
        if (p->Int())
          ret.first = Type::Spec::kInt;
        else if (p->Void())
          ret.first = Type::Spec::kVoid;
        // else if ()  后续添加其他类型的说明符，比如double float等
        else
          ABORT(); // 未知的类型说明符
      } 
      else
        ABORT(); // 未知的类型说明符
    }
    else
      ABORT();
  }

  return ret;
}

std::pair<TypeExpr*, std::string>
Ast2Asg::operator()(ast::DeclaratorContext* ctx, TypeExpr* sub)
{
  return self(ctx->directDeclarator(), sub);
}

static int
eval_arrlen(Expr* expr)
{
  if (auto p = expr->dcst<IntegerLiteral>())
    return p->val;

  if (auto p = expr->dcst<DeclRefExpr>()) {
    if (p->decl == nullptr)
      ABORT();

    auto var = p->decl->dcst<VarDecl>();
    if (!var || !var->type->qual.const_)
      ABORT(); // 数组长度必须是编译期常量

    switch (var->type->spec) {
      case Type::Spec::kChar:
      case Type::Spec::kInt:
      case Type::Spec::kLong:
      case Type::Spec::kLongLong:
        return eval_arrlen(var->init);

      default:
        ABORT(); // 长度表达式必须是数值类型
    }
  }

  if (auto p = expr->dcst<UnaryExpr>()) {
    auto sub = eval_arrlen(p->sub);

    switch (p->op) {
      case UnaryExpr::kPos:
        return sub;

      case UnaryExpr::kNeg:
        return -sub;

      case UnaryExpr::kNot:
      default:
        ABORT();
    }
  }

  if (auto p = expr->dcst<BinaryExpr>()) {
    auto lft = eval_arrlen(p->lft);
    auto rht = eval_arrlen(p->rht);

    switch (p->op) {
      case BinaryExpr::kAdd:
        return lft + rht;

      case BinaryExpr::kSub:
        return lft - rht;

      default:
        ABORT();
    }
  }

  if (auto p = expr->dcst<InitListExpr>()) {
    if (p->list.empty())
      return 0;
    return eval_arrlen(p->list[0]);
  }

  ABORT();
}

std::pair<TypeExpr*, std::string>
Ast2Asg::operator()(ast::DirectDeclaratorContext* ctx, TypeExpr* sub)
{
  if (auto p = ctx->Identifier())
    return { sub, p->getText() };

  if (ctx->LeftBracket()) {
    auto arrayType = make<ArrayType>();
    arrayType->sub = sub;

    if (auto p = ctx->assignmentExpression())
      arrayType->len = eval_arrlen(self(p));
    else
      arrayType->len = ArrayType::kUnLen;

    return self(ctx->directDeclarator(), arrayType);
  }

  ABORT();
}

//==============================================================================
// 表达式
//==============================================================================

Expr*
Ast2Asg::operator()(ast::ExpressionContext* ctx)
{
  auto list = ctx->assignmentExpression();
  Expr* ret = self(list[0]);

  for (unsigned i = 1; i < list.size(); ++i) {
    auto node = make<BinaryExpr>();
    node->op = node->kComma;
    node->lft = ret;
    node->rht = self(list[i]);
    ret = node;
  }

  return ret;
}

Expr*
Ast2Asg::operator()(ast::AssignmentExpressionContext* ctx)
{
  if (auto p = ctx->orExpression())
    return self(p);

  auto ret = make<BinaryExpr>();
  ret->op = ret->kAssign;
  ret->lft = self(ctx->unaryExpression());
  ret->rht = self(ctx->assignmentExpression());
  return ret;
}

// 乘法表达式
Expr*
Ast2Asg::operator()(ast::MultiplicativeExpressionContext* ctx)
{
  auto children = ctx->children;  // 获取当前节点及上下文中的所有子节点列表
  Expr* ret = self(dynamic_cast<ast::UnaryExpressionContext*>(children[0])); // 初始化当前节点的处理结果

  for (unsigned i = 1; i < children.size(); ++i) {  // 如果存在子节点
    auto node = make<BinaryExpr>(); // 创建新的BinaryExpr对象node，用来表达二元表达式

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i]) // 获取当前节点并转换成叶子结点类型
                   ->getSymbol()  // 返回与诊断节点关联的符号
                   ->getType();   // 返回表示类型的整数·
    switch (token) {  // 决定当前节点操作
      case ast::Star:
        node->op = node->kMul; 
        break;

      case ast::Slash:
        node->op = node->kDiv;
        break;

      case ast::Percent:
        node->op = node->kMod;
        break;
      
      default:
        ABORT();  // 未知操作符，编译错误
    }

    node->lft = ret;
    // 将下一个子节点强制转换成UnaryExpression类型，然后递归调用self，以解析这个子表达式并获取其计算结果
    node->rht = self(dynamic_cast<ast::UnaryExpressionContext*>(children[++i]));  
    ret = node;
  }

  return ret;
}


// 加法表达式
Expr*
Ast2Asg::operator()(ast::AdditiveExpressionContext* ctx)
{
  auto children = ctx->children;  // 获取当前节点及上下文中的所有子节点列表
  Expr* ret = self(dynamic_cast<ast::MultiplicativeExpressionContext*>(children[0])); // 初始化当前节点的处理结果

  for (unsigned i = 1; i < children.size(); ++i) {  // 如果存在子节点
    auto node = make<BinaryExpr>(); // 创建新的BinaryExpr对象node，用来表达二元表达式

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i]) // 获取当前节点并转换成叶子结点类型
                   ->getSymbol()  // 返回与诊断节点关联的符号
                   ->getType();   // 返回表示类型的整数·
    switch (token) {  // 决定当前节点是加法操作还是减法操作
      case ast::Plus:
        node->op = node->kAdd; 
        break;

      case ast::Minus:
        node->op = node->kSub;
        break;

      default:
        ABORT();  // 未知操作符，编译错误
    }

    node->lft = ret;
    // 将下一个子节点强制转换成UnaryExpression类型，然后递归调用self，以解析这个子表达式并获取其计算结果
    node->rht = self(dynamic_cast<ast::MultiplicativeExpressionContext*>(children[++i]));  
    ret = node;
  }

  return ret;
}

// 关系表达式
Expr*
Ast2Asg::operator()(ast::RelationalExpressionContext* ctx)
{
  auto children = ctx->children;  // 获取当前节点及上下文中的所有子节点列表
  Expr* ret = self(dynamic_cast<ast::AdditiveExpressionContext*>(children[0])); // 初始化当前节点的处理结果

  for (unsigned i = 1; i < children.size(); ++i) {  // 如果存在子节点
    auto node = make<BinaryExpr>(); // 创建新的BinaryExpr对象node，用来表达二元表达式

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i]) // 获取当前节点并转换成叶子结点类型
                   ->getSymbol()  // 返回与诊断节点关联的符号
                   ->getType();   // 返回表示类型的整数·
    switch (token) {  // 决定当前节点操作
      case ast::Greater:
        node->op = node->kGt; 
        break;

      case ast::Less:
        node->op = node->kLt;
        break;

      case ast::Greaterequal:
        node->op = node->kGe;
        break;

      case ast::Lessequal:
        node->op = node->kLe;
        break;

      default:
        ABORT();  // 未知操作符，编译错误
    }

    node->lft = ret;
    node->rht = self(dynamic_cast<ast::AdditiveExpressionContext*>(children[++i]));  
    ret = node;
  }

  return ret;

}



// 相等表达式
Expr* 
Ast2Asg::operator()(ast::EqualityExpressionContext* ctx)
{
  auto children = ctx->children;  // 获取当前节点及上下文中的所有子节点列表
  Expr* ret = self(dynamic_cast<ast::RelationalExpressionContext*>(children[0])); // 初始化当前节点的处理结果

  for (unsigned i = 1; i < children.size(); ++i) {  // 如果存在子节点
    auto node = make<BinaryExpr>(); // 创建新的BinaryExpr对象node，用来表达二元表达式

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i]) // 获取当前节点并转换成叶子结点类型
                   ->getSymbol()  // 返回与诊断节点关联的符号
                   ->getType();   // 返回表示类型的整数·
    switch (token) {  // 决定当前节点操作
      case ast::Equalequal:
        node->op = node->kEq; 
        break;

      case ast::Exclaimequal:
        node->op = node->kNe;
        break;

      default:
        ABORT();  // 未知操作符，编译错误
    }

    node->lft = ret;
    node->rht = self(dynamic_cast<ast::RelationalExpressionContext*>(children[++i]));  
    ret = node;
  }

  return ret;

}

// 逻辑与表达式
Expr*
Ast2Asg::operator()(ast::AndExpressionContext* ctx)
{
  auto children = ctx->children;  // 获取当前节点及上下文中的所有子节点列表
  Expr* ret = self(dynamic_cast<ast::EqualityExpressionContext*>(children[0])); // 初始化当前节点的处理结果

  for (unsigned i = 1; i < children.size(); ++i) {  // 如果存在子节点
    auto node = make<BinaryExpr>(); // 创建新的BinaryExpr对象node，用来表达二元表达式

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i]) // 获取当前节点并转换成叶子结点类型
                   ->getSymbol()  // 返回与诊断节点关联的符号
                   ->getType();   // 返回表示类型的整数·
    switch (token) {  // 决定当前节点操作
      case ast::Ampamp:
        node->op = node->kAnd; 
        break;

      default:
        ABORT();  // 未知操作符，编译错误
    }

    node->lft = ret;
    node->rht = self(dynamic_cast<ast::EqualityExpressionContext*>(children[++i]));  
    ret = node;
  }

  return ret;
}

// 逻辑或表达式
Expr*
Ast2Asg::operator()(ast::OrExpressionContext* ctx)
{
  auto children = ctx->children;  // 获取当前节点及上下文中的所有子节点列表
  Expr* ret = self(dynamic_cast<ast::AndExpressionContext*>(children[0])); // 初始化当前节点的处理结果

  for (unsigned i = 1; i < children.size(); ++i) {  // 如果存在子节点
    auto node = make<BinaryExpr>(); // 创建新的BinaryExpr对象node，用来表达二元表达式

    auto token = dynamic_cast<antlr4::tree::TerminalNode*>(children[i]) // 获取当前节点并转换成叶子结点类型
                   ->getSymbol()  // 返回与诊断节点关联的符号
                   ->getType();   // 返回表示类型的整数·
    switch (token) {  // 决定当前节点操作
      case ast::Pipepipe:
        node->op = node->kOr; 
        break;

      default:
        ABORT();  // 未知操作符，编译错误
    }

    node->lft = ret;
    node->rht = self(dynamic_cast<ast::AndExpressionContext*>(children[++i]));  
    ret = node;
  }

  return ret;
}


Expr*
Ast2Asg::operator()(ast::UnaryExpressionContext* ctx)
{
  if (auto p = ctx->postfixExpression())
    return self(p);

  auto ret = make<UnaryExpr>();

  switch (
    dynamic_cast<antlr4::tree::TerminalNode*>(ctx->unaryOperator()->children[0])
      ->getSymbol()
      ->getType()) {
    case ast::Plus:
      ret->op = ret->kPos;
      break;

    case ast::Minus:
      ret->op = ret->kNeg;
      break;

    case ast::Exclaim:
      ret->op = ret->kNot;
      break;

    default:
      ABORT();
  }

  ret->sub = self(ctx->unaryExpression());

  return ret;
}

Expr*
Ast2Asg::operator()(ast::PostfixExpressionContext* ctx)
{
  if (auto p = ctx->primaryExpression()){
    return self(p);
  }
  // 数组情况
  if (ctx->expression()){
    auto ret = make<BinaryExpr>();
    ret->op = ret->kIndex;
    ret->lft = self(ctx->postfixExpression());
    ret->rht = self(ctx->expression());
    return ret;
  }

  // 函数调用
  if (ctx->LeftParen()){
    auto ret = make<CallExpr>();
    ret->head = self(ctx->postfixExpression());
    if(auto p = ctx->argumentExpressionList()){
      for(auto&& i : p->assignmentExpression()){
        ret->args.push_back(self(i));
      }
    }
    return ret;
  }
  ABORT();
}

Expr*
Ast2Asg::operator()(ast::PrimaryExpressionContext* ctx)
{

  if (auto p = ctx->Identifier()) {
    auto name = p->getText();
    auto ret = make<DeclRefExpr>();
    ret->decl = mSymtbl->resolve(name);
    return ret;
  }

  if (auto p = ctx->Constant()) {
    auto text = p->getText();

    auto ret = make<IntegerLiteral>();

    ASSERT(!text.empty());
    if (text[0] != '0')
      ret->val = std::stoll(text);

    else if (text.size() == 1)
      ret->val = 0;

    else if (text[1] == 'x' || text[1] == 'X')
      ret->val = std::stoll(text.substr(2), nullptr, 16);

    else
      ret->val = std::stoll(text.substr(1), nullptr, 8);

    return ret;
  }

  // 处理带括号的基本表达式
  if (auto p = ctx->expression()) {
    auto ret = make<ParenExpr>();
    ret->sub = self(p);
    return ret;
  }
  ABORT();
}

Expr*
Ast2Asg::operator()(ast::InitializerContext* ctx)
{
  if (auto p = ctx->assignmentExpression())
    return self(p);

  auto ret = make<InitListExpr>();

  if (auto p = ctx->initializerList()) {
    for (auto&& i : p->initializer()) {
      // 将初始化列表展平
      auto expr = self(i);
      if (auto p = expr->dcst<InitListExpr>()) {
        for (auto&& sub : p->list)
          ret->list.push_back(sub);
      } else {
        ret->list.push_back(expr);
      }
    }
  }

  return ret;
}

//==============================================================================
// 语句
//==============================================================================

Stmt*
Ast2Asg::operator()(ast::StatementContext* ctx)
{
  if (auto p = ctx->compoundStatement())
    return self(p);

  if (auto p = ctx->expressionStatement())
    return self(p);

  if (auto p = ctx->jumpStatement())
    return self(p);

  if (auto p = ctx->ifStatement())
    return self(p);
  
  if (auto p = ctx->whileStatement())
    return self(p);

  if (auto p = ctx->breakStatement())
    return self(p);
  
  if (auto p = ctx->continueStatement())
    return self(p);
  ABORT();
}

CompoundStmt*
Ast2Asg::operator()(ast::CompoundStatementContext* ctx)
{
  auto ret = make<CompoundStmt>();

  if (auto p = ctx->blockItemList()) {
    Symtbl localDecls(self);

    for (auto&& i : p->blockItem()) {
      if (auto q = i->declaration()) {
        auto sub = make<DeclStmt>();
        sub->decls = self(q);
        ret->subs.push_back(sub);
      }

      else if (auto q = i->statement())
        ret->subs.push_back(self(q));

      else
        ABORT();
    }
  }

  return ret;
}

Stmt*
Ast2Asg::operator()(ast::ExpressionStatementContext* ctx)
{
  if (auto p = ctx->expression()) {
    auto ret = make<ExprStmt>();
    ret->expr = self(p);
    return ret;
  }

  return make<NullStmt>();
}

Stmt*
Ast2Asg::operator()(ast::JumpStatementContext* ctx)
{
  if (ctx->Return()) {
    auto ret = make<ReturnStmt>();
    ret->func = mCurrentFunc;
    if (auto p = ctx->expression())
      ret->expr = self(p);
    return ret;
  }

  ABORT();
}

// if 语句
IfStmt* 
Ast2Asg::operator() (ast::IfStatementContext* ctx)
{
  auto ret = make<IfStmt>();
  ret->cond = self(ctx->expression());
  ret->then = self(ctx->statement());
  if (auto p = ctx->elseStatement())
    ret->else_ = self(p);

  return ret;
}

Stmt* 
Ast2Asg::operator() (ast::ElseStatementContext* ctx)
{
  if(auto p = ctx->statement()){
    return self(p);
  }
  ABORT();
}

// while 语句
WhileStmt*
Ast2Asg::operator() (ast::WhileStatementContext* ctx)
{
  auto ret = make<WhileStmt>();
  ret->cond = self(ctx->expression());
  ret->body = self(ctx->statement());
  return ret;
}

// Break 语句
BreakStmt* 
Ast2Asg::operator() (ast::BreakStatementContext* ctx)
{
  auto ret = make<BreakStmt>();
  return ret;
}

// Conutinue 语句
ContinueStmt* 
Ast2Asg::operator() (ast::ContinueStatementContext* ctx)
{
  auto ret = make<ContinueStmt>();
  return ret;
}

//==============================================================================
// 声明
//==============================================================================

std::vector<Decl*>
Ast2Asg::operator()(ast::DeclarationContext* ctx)
{
  std::vector<Decl*> ret;

  auto specs = self(ctx->declarationSpecifiers());

  if (auto p = ctx->initDeclaratorList()) {
    for (auto&& j : p->initDeclarator())
      ret.push_back(self(j, specs));
  }

  // 如果 initDeclaratorList 为空则这行声明语句无意义
  return ret;
}

FunctionDecl*
Ast2Asg::operator()(ast::FunctionDefinitionContext* ctx) 
{
  auto ret = make<FunctionDecl>();
  mCurrentFunc = ret;

  auto type = make<Type>();
  ret->type = type;

  auto sq = self(ctx->declarationSpecifiers());
  type->spec = sq.first, type->qual = sq.second;

  auto [texp, name] = self(ctx->directDeclarator(), nullptr);
  auto funcType = make<FunctionType>();
  funcType->sub = texp;
  type->texp = funcType;
  ret->name = std::move(name);

  // 创建局部符号表
  Symtbl localDecls(*this);

  // 将函数自身先加入符号表，以支持递归调用
  (*mSymtbl)[ret->name] = ret;

  if (auto p = ctx->functionArguments()) {
    for (auto&& i : p->funcArgsDeclaration()) {
      auto vdecl = make<VarDecl>();

      // 确定参数类型
      auto param_type = make<Type>();
      vdecl->type = param_type;
      auto sq_param = self(i->declarationSpecifiers());
      param_type->spec = sq_param.first;
      param_type->qual = sq_param.second;

      // 确定参数名
      auto [texp, name] = self(i->directDeclarator(), nullptr);
      param_type->texp = texp;
      vdecl->name = std::move(name);

      // 将参数添加到函数参数列表
      ret->params.push_back(vdecl);

      // 添加参数到局部符号表
      localDecls[vdecl->name] = vdecl;
    }
  }

  ret->body = self(ctx->compoundStatement());

  return ret;
}


Decl*
Ast2Asg::operator()(ast::InitDeclaratorContext* ctx, SpecQual sq)
{

  Decl* ret;

  if (auto p = ctx->functionDeclaration()) {
    auto funcDecl = make<FunctionDecl>();
    auto func_type = make<Type>();
    funcDecl->type = func_type;

    // 函数声明的类型
    func_type->spec = sq.first;
    func_type->qual = sq.second;

    auto [unused, name] = self(p->directDeclarator(), nullptr);

    // 函数类型和函数名
    func_type->texp = make<FunctionType>();
    funcDecl->name = std::move(name);

    // 函数参数
    if (auto q = p->parameterTypeList()) {
      for (auto&& i : q->parameterDeclaration()) {
        auto vdecl = make<VarDecl>();
        auto param_type = make<Type>();
        vdecl->type = param_type;

        // 确定参数类型
        auto sq_param = self(i->declarationSpecifiers());
        param_type->spec = sq_param.first;
        param_type->qual = sq_param.second;

        // 确定参数名和类型表达式
        auto [texp, name] = self(i->declarator(), nullptr);
        param_type->texp = texp;
        vdecl->name = std::move(name);

        funcDecl->params.push_back(vdecl);
      }
    }

    funcDecl->body = nullptr;
    ret = funcDecl;
  }

  else {
    auto [texp, name] = self(ctx->declarator(), nullptr);
    auto vdecl = make<VarDecl>();
    auto type = make<Type>();
    vdecl->type = type;

    type->spec = sq.first;
    type->qual = sq.second;
    type->texp = texp;
    vdecl->name = std::move(name);

    if (auto p = ctx->initializer())
      vdecl->init = self(p);
    else
      vdecl->init = nullptr;

    ret = vdecl;
  }

  // 这个实现允许符号重复定义，新定义会取代旧定义
  (*mSymtbl)[ret->name] = ret;
  return ret;
}

} // namespace asg
