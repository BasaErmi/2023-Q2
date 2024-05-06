parser grammar SYsUParser;

options {
  tokenVocab=SYsULexer;
}

primaryExpression   // 基本表达式
    :   Identifier
    |   Constant
    |   LeftParen expression RightParen
    ;   

postfixExpression   // 后缀表达式
    :   primaryExpression  
    |   postfixExpression LeftBracket expression RightBracket
    |   postfixExpression LeftParen argumentExpressionList? RightParen
    ;

argumentExpressionList  // 参数表达式列表
    :   assignmentExpression (Comma assignmentExpression)*
    ;

unaryExpression     // 一元表达式，可以是一个后缀表达式或者一个一元操作符+一元表达式
    :
    (postfixExpression
    |   unaryOperator unaryExpression
    )
    ;

unaryOperator   //  一元操作符
    :   Plus 
    |   Minus
    |   Exclaim
    ;

multiplicativeExpression    // 乘法表达式
    :   unaryExpression ((Star|Percent|Slash) unaryExpression)*
    ;

additiveExpression  // 加法表达式
    :   multiplicativeExpression ((Plus|Minus) multiplicativeExpression)*
    ;

relationalExpression  // 关系表达式
    :   additiveExpression ((Greater|Less|Greaterequal|Lessequal) additiveExpression)*
    ;

equalityExpression  // 相等性表达式
    :   relationalExpression ((Equalequal|Exclaimequal) relationalExpression)*
    ;

andExpression
    :   equalityExpression (Ampamp equalityExpression)* // 逻辑与表达式
    ;

orExpression
    :   andExpression (Pipepipe andExpression)* // 逻辑或表达式
    ;

assignmentExpression    // 赋值表达式
    :   orExpression
    |   unaryExpression Equal assignmentExpression
    ;

expression  // 表达式，它是一个赋值表达式后跟零个或多个由逗号（Comma）分隔的赋值表达式
    :   assignmentExpression (Comma assignmentExpression)*
    ;


declaration // 声明规范后跟一个可选的初始化声明列表（initDeclaratorList）和分号
    :   declarationSpecifiers initDeclaratorList? Semi
    ;

declarationSpecifiers   // 声明规范列表
    :   declarationSpecifier+
    ;

declarationSpecifier    // 声明规范
    :   typeSpecifier 
    |   Const typeSpecifier
    ;

initDeclaratorList  // 初始化声明列表，由一个或多个用逗号分隔的初始化声明组成
    :   initDeclarator (Comma initDeclarator)*
    ;

initDeclarator  // 初始化声明器，后面跟一个可选的等号后加初始化器
    :   declarator (Equal initializer)?
    |   functionDeclaration
    ;

functionDeclaration
    :   directDeclarator LeftParen parameterTypeList? RightParen  // 函数参数列表
    ;

parameterTypeList
    :   parameterDeclaration (Comma parameterDeclaration)*
    ;

parameterDeclaration
    :   declarationSpecifiers declarator
    |   declarationSpecifiers  // 无参数名的声明（比如函数原型中的）
    ;


typeSpecifier   // 类型规范，可能需要修改
    :   Int
    |   Void
    ;


declarator  // 声明器
    :   directDeclarator
    ;

directDeclarator    // 直接声明器，声明整个变量、数组或函数
    :   Identifier  // 变量名
    |   directDeclarator LeftBracket assignmentExpression? RightBracket  // 数组大小声明
    ;




identifierList  //初始化列表，比如说int a, b, c，这样，[a, b, c]就是identifierList
    :   Identifier (Comma Identifier)*
    ;

initializer     // 初始化，比如 = a = 3或者 = {1, 2, 3, 4}
    :   assignmentExpression    
    |   LeftBrace initializerList? Comma? RightBrace
    ;

initializerList // 初始化列表   
    // :   designation? initializer (Comma designation? initializer)*
    :   initializer (Comma initializer)*
    ;

statement // 语句
    :   compoundStatement
    |   expressionStatement
    |   jumpStatement
    |   ifStatement
    |   whileStatement
    |   breakStatement
    |   continueStatement
    ;

compoundStatement   // 复合语句，是若干个语句列表的组合外加花括号
    :   LeftBrace blockItemList? RightBrace
    ;

blockItemList // 语句列表，是一个个语句的列表
    :   blockItem+
    ;

blockItem   // 语句项，是一个语句或者声明
    :   statement
    |   declaration
    ;

expressionStatement // 表达式语句
    :   expression? Semi
    ;

jumpStatement // 跳转语句，返回语句（Return）后跟一个可选的表达式（expression）和一个分号（Semi）。
    :   (Return expression?)
    Semi
    ;

compilationUnit // 编译单元
    :   translationUnit? EOF
    ;

translationUnit  // 翻译单元，一个或多个外部声明的列表
    :   externalDeclaration+
    ;

externalDeclaration // 外部声明，是函数定义或者声明
    :   functionDefinition      // 函数定义
    |   declaration             // 声明
    ;

functionDefinition //函数定义： 声明规范 直接声明器（一个变量或者带括号数组） ()  复合语句
    : declarationSpecifiers directDeclarator LeftParen  functionArguments? RightParen compoundStatement
    ;

functionArguments   // 函数参数，由一个或多个用逗号分隔的声明组成
    :   funcArgsDeclaration (Comma funcArgsDeclaration)*
    ;    

funcArgsDeclaration 
    :   declarationSpecifiers directDeclarator
    ;

ifStatement
    :   If LeftParen expression RightParen statement elseStatement?
    ;


elseStatement
    : Else statement
    ;

whileStatement
    : While LeftParen expression RightParen statement
    ;

breakStatement
    : Break Semi
    ;

continueStatement
    : Continue Semi
    ;

    