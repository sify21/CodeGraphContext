"""
Tree-sitter 查询与 RPC 识别：Java 中的 Spring HTTP、OpenFeign、gRPC。

JAVA_RPC_QUERIES 用于识别：
- Spring 定义的 HTTP 接口（Controller 类及其端点方法）
- spring-cloud-openfeign 定义的 HTTP 调用（@FeignClient 接口及其方法）
- gRPC 定义的服务接口（继承 *Grpc.ServiceImplBase 等的类）
- gRPC 服务端点方法（ServiceImplBase 子类中的 rpc 实现函数）
- gRPC 客户端调用（通过 *Stub 调用的方法）
- gRPC 生成 Stub（Stub 类及其方法定义，便于识别“生成代码入库”的场景）
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tree_sitter import Node

from codegraphcontext.utils.tree_sitter_manager import execute_query

# ---------------------------------------------------------------------------
# Spring 常用注解名（简单名，用于过滤）
# ---------------------------------------------------------------------------
SPRING_CONTROLLER_ANNOTATIONS = frozenset({"RestController", "Controller"})
SPRING_MAPPING_ANNOTATIONS = frozenset({
    "GetMapping", "PostMapping", "PutMapping", "DeleteMapping",
    "RequestMapping", "PatchMapping",
})
FEIGN_CLIENT_ANNOTATION = "FeignClient"

# gRPC 服务端：父类名通常包含 Grpc 或 ServiceImplBase / BindableService
GRPC_SERVICE_SUPERCLASS_HINTS = ("Grpc", "ServiceImplBase", "BindableService")
# gRPC 客户端：调用对象名通常以 Stub 结尾
GRPC_CLIENT_STUB_SUFFIX = "Stub"
# gRPC 服务类中常见但通常不视为业务 RPC 端点的方法名
GRPC_NON_ENDPOINT_METHOD_NAMES = frozenset({
    "bindService",
    "equals",
    "hashCode",
    "toString",
})
# gRPC 生成 Stub：父类名中常见关键字（AbstractAsyncStub / AbstractBlockingStub / ...）
GRPC_STUB_SUPERCLASS_HINTS = (
    "AbstractStub",
    "AbstractAsyncStub",
    "AbstractBlockingStub",
    "AbstractFutureStub",
)
# Stub 类中常见基础方法（通常不是具体业务 RPC）
GRPC_STUB_NON_RPC_METHOD_NAMES = frozenset({
    "build",
    "newStub",
    "newFutureStub",
    "newBlockingStub",
    "equals",
    "hashCode",
    "toString",
})


def _annotation_simple_name(text: str) -> str:
    """从完整注解名（可能带包路径）取简单名。"""
    if not text:
        return ""
    return text.split(".")[-1].strip()


def _get_node_text(node: Any) -> str:
    if node is None:
        return ""
    return node.text.decode("utf-8", errors="replace")


def _find_ancestor(node: Any, *types: str) -> Optional[Any]:
    curr = node.parent
    while curr:
        if curr.type in types:
            return curr
        curr = curr.parent
    return None


def _has_override_annotation(method_node: Any, get_text: Any) -> bool:
    """判断方法是否带 @Override 注解。"""
    for child in method_node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type not in ("marker_annotation", "annotation"):
                continue
            ann_name_node = mod.child_by_field_name("name")
            if ann_name_node and _annotation_simple_name(get_text(ann_name_node)) == "Override":
                return True
    return False


def _is_grpc_endpoint_method(
    method_node: Any, method_name: str, params_text: str, get_text: Any
) -> Tuple[bool, str]:
    """
    分层判定 gRPC 服务端点方法：
    1) 参数含 StreamObserver（经典 grpc-java）
    2) 带 @Override 且非常见辅助方法
    3) 返回类型呈现 Mono/Flux/Flow（常见 reactive/coroutine 风格代码特征）
    """
    if "StreamObserver" in params_text:
        return True, "stream_observer_param"

    if method_name and method_name in GRPC_NON_ENDPOINT_METHOD_NAMES:
        return False, ""

    if _has_override_annotation(method_node, get_text):
        return True, "override_in_grpc_service"

    return_type_node = method_node.child_by_field_name("type")
    return_type = get_text(return_type_node) if return_type_node else ""
    if any(x in return_type for x in ("Mono", "Flux", "Flow")):
        return True, "reactive_return_type"

    return False, ""


def _is_grpc_stub_class(class_node: Any, get_text: Any) -> Tuple[bool, str]:
    """判断 class 是否为 gRPC 生成 Stub 类。返回 (is_stub, reason)。"""
    name_node = class_node.child_by_field_name("name")
    class_name = get_text(name_node) if name_node else ""
    superclass_node = class_node.child_by_field_name("superclass")
    superclass_text = get_text(superclass_node) if superclass_node else ""

    outer_class = _find_ancestor(class_node, "class_declaration")
    outer_name = ""
    if outer_class:
        outer_name_node = outer_class.child_by_field_name("name")
        outer_name = get_text(outer_name_node) if outer_name_node else ""

    by_superclass = any(hint in superclass_text for hint in GRPC_STUB_SUPERCLASS_HINTS)
    by_name = class_name.endswith(GRPC_CLIENT_STUB_SUFFIX)
    by_outer = outer_name.endswith("Grpc")

    # 更严格：必须具备 Abstract*Stub 超类线索，再结合命名/外层上下文
    # 典型 generated stub：GreeterGrpc.GreeterStub extends AbstractAsyncStub<...>
    if by_superclass and by_name and by_outer:
        return True, "stub_name_superclass_and_grpc_outer"
    if by_superclass and by_name:
        return True, "stub_name_and_superclass"
    if by_superclass and by_outer:
        return True, "stub_superclass_in_grpc_outer_class"
    return False, ""


# 映射注解简单名 -> HTTP 方法（用于 GetMapping 等）
_MAPPING_ANNOTATION_TO_METHOD = {
    "GetMapping": "GET",
    "PostMapping": "POST",
    "PutMapping": "PUT",
    "DeleteMapping": "DELETE",
    "PatchMapping": "PATCH",
    "RequestMapping": "",  # 需从 method= 解析或默认 GET
}


def _normalize_http_method(method: str) -> str:
    """将 RequestMethod.GET、GET 等规范为 GET。"""
    if not method:
        return "GET"
    s = method.strip().upper()
    if "." in s:
        s = s.split(".")[-1]
    return s if s in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS") else "GET"


def _string_literal_content(text: str) -> str:
    """去掉 Java 字符串字面量两侧引号。"""
    if not text:
        return ""
    t = text.strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("'") and t.endswith("'")):
        return t[1:-1]
    return t


def _extract_annotation_path_and_method(
    ann_node: Any, annotation_simple_name: str, get_text: Any
) -> Tuple[str, str]:
    """
    从方法上的映射注解（GetMapping / PostMapping / RequestMapping 等）解析 path 与 HTTP method。
    返回 (path, http_method)，path 不含类级前缀。
    """
    path = ""
    method = _MAPPING_ANNOTATION_TO_METHOD.get(annotation_simple_name, "")
    if ann_node.type == "marker_annotation":
        return path, method or "GET"
    # annotation 带 argument_list
    arg_list = ann_node.child_by_field_name("argument_list")
    if not arg_list:
        return path, method or "GET"
    for child in arg_list.children:
        if child.type != "element_value_pair":
            # 可能是单元素 value，如 @GetMapping("/x")
            if child.type == "string_literal":
                path = _string_literal_content(get_text(child))
            continue
        key_node = child.child_by_field_name("name")
        if key_node is None:
            key_node = child.child_by_field_name("key")
        key = get_text(key_node).strip() if key_node else ""
        val_node = child.child_by_field_name("value")
        if not val_node:
            continue
        val_text = get_text(val_node).strip()
        if key in ("value", "path"):
            path = _string_literal_content(val_text)
        elif key == "method":
            method = _normalize_http_method(val_text)
    if not method and annotation_simple_name == "RequestMapping":
        method = "GET"
    return path, method or "GET"


def _get_class_level_request_mapping_path(class_node: Any, get_text: Any) -> str:
    """从类上的 @RequestMapping 取 value/path，用于拼接端点完整 path。"""
    for child in class_node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type not in ("marker_annotation", "annotation"):
                continue
            name_node = mod.child_by_field_name("name")
            if not name_node:
                continue
            name = _annotation_simple_name(get_text(name_node))
            if name != "RequestMapping":
                continue
            p, _ = _extract_annotation_path_and_method(mod, "RequestMapping", get_text)
            return p or ""
    return ""


def _join_path(class_path: str, method_path: str) -> str:
    """拼接类级 path 与方法级 path，保证中间只有一个 /。"""
    c = (class_path or "").strip().strip("/")
    m = (method_path or "").strip().strip("/")
    if not c:
        return "/" + m if m else ""
    if not m:
        return "/" + c
    return "/" + c + "/" + m


def _extract_feign_client_base(interface_node: Any, get_text: Any) -> Tuple[str, str]:
    """从 @FeignClient 注解解析 url= 与 name=，用于与 Controller 关联。返回 (base_url, service_name)。"""
    base_url = ""
    service_name = ""
    for child in interface_node.children:
        if child.type != "modifiers":
            continue
        for mod in child.children:
            if mod.type != "annotation":
                continue
            name_node = mod.child_by_field_name("name")
            if not name_node:
                continue
            if _annotation_simple_name(get_text(name_node)) != FEIGN_CLIENT_ANNOTATION:
                continue
            arg_list = mod.child_by_field_name("argument_list")
            if not arg_list:
                break
            for arg in arg_list.children:
                if arg.type != "element_value_pair":
                    continue
                key_node = arg.child_by_field_name("name") or arg.child_by_field_name("key")
                key = get_text(key_node).strip() if key_node else ""
                val_node = arg.child_by_field_name("value")
                if not val_node:
                    continue
                val_text = get_text(val_node).strip()
                if key == "url":
                    base_url = _string_literal_content(val_text)
                elif key == "name":
                    service_name = _string_literal_content(val_text)
            break
    return base_url, service_name


# ---------------------------------------------------------------------------
# Tree-sitter 查询：仅做结构匹配，具体框架由上面常量 + 下方解析逻辑过滤
# ---------------------------------------------------------------------------

JAVA_RPC_QUERIES = {
    # 1) 使用 Spring 定义的 HTTP 接口：带注解的 Controller 类
    #    匹配带有任意注解的 class，解析时过滤 RestController / Controller
    "spring_http_controller": """
        (class_declaration
            (modifiers
                [
                    (marker_annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                    (annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                ])
            name: (identifier) @class_name
        ) @controller_class
    """,

    # 2) Spring HTTP 端点方法：带映射注解的方法（GetMapping / PostMapping / ...）
    "spring_http_endpoint_method": """
        (method_declaration
            (modifiers
                [
                    (marker_annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                    (annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                ])
            name: (identifier) @method_name
            parameters: (formal_parameters) @params
        ) @endpoint_method
    """,

    # 3) 使用 spring-cloud-openfeign 定义的 HTTP 调用：@FeignClient 接口
    "feign_client_interface": """
        (interface_declaration
            (modifiers
                [
                    (marker_annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                    (annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                ])
            name: (identifier) @interface_name
        ) @feign_interface
    """,

    # 4) Feign 接口中的方法（即 HTTP 调用函数定义）
    "feign_client_method": """
        (interface_declaration
            (modifiers
                [
                    (marker_annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                    (annotation name: [
                        (identifier)
                        (scoped_type_identifier)
                    ] @ann_name)
                ])
            body: (interface_body
                (method_declaration
                    name: (identifier) @method_name
                    parameters: (formal_parameters) @params
                ) @feign_method
            )
        ) @feign_interface
    """,

    # 5) 通过 gRPC 定义的服务接口：继承 *Grpc.ServiceImplBase 等的类
    "grpc_service_class": """
        (class_declaration
            name: (identifier) @class_name
            superclass: [
                (type_identifier)
                (scoped_type_identifier)
                (generic_type)
            ] @superclass
        ) @grpc_service_class
    """,

    # 6) gRPC 服务端点方法：位于 gRPC 服务实现类中的 method_declaration
    "grpc_service_endpoint_method": """
        (method_declaration
            name: (identifier) @method_name
            parameters: (formal_parameters) @params
        ) @grpc_service_endpoint
    """,

    # 7) 通过 gRPC client 调用服务的函数：*Stub.xxx() 形式的方法调用 (应对Stub类在编译时生成的情况)
    "grpc_client_call": """
        (method_invocation
            object: [
                (identifier) @receiver
                (field_access
                    field: (identifier) @receiver
                )
            ]
            name: (identifier) @method_name
        ) @grpc_call
    """,

    # 8) gRPC 生成 Stub 类：更严格匹配（优先内嵌在 *Grpc 外部类且带 superclass）
    "grpc_stub_class": """
        [
            (class_declaration
                name: (identifier) @outer_class_name
                (class_body
                    (class_declaration
                        name: (identifier) @class_name
                        superclass: [
                            (type_identifier)
                            (scoped_type_identifier)
                            (generic_type)
                        ] @superclass
                    ) @grpc_stub_class
                )
            )
            (class_declaration
                name: (identifier) @class_name
                superclass: [
                    (type_identifier)
                    (scoped_type_identifier)
                    (generic_type)
                ] @superclass
            ) @grpc_stub_class
        ]
    """,

    # 9) gRPC 生成 Stub 方法：更严格匹配（方法需位于带 superclass 的类内）
    "grpc_stub_method": """
        [
            (class_declaration
                name: (identifier) @outer_class_name
                (class_body
                    (class_declaration
                        name: (identifier) @class_name
                        superclass: [
                            (type_identifier)
                            (scoped_type_identifier)
                            (generic_type)
                        ] @superclass
                        (class_body
                            (method_declaration
                                name: (identifier) @method_name
                                parameters: (formal_parameters) @params
                            ) @grpc_stub_method
                        )
                    )
                )
            )
            (class_declaration
                name: (identifier) @class_name
                superclass: [
                    (type_identifier)
                    (scoped_type_identifier)
                    (generic_type)
                ] @superclass
                (class_body
                    (method_declaration
                        name: (identifier) @method_name
                        parameters: (formal_parameters) @params
                    ) @grpc_stub_method
                )
            )
        ]
    """,
}


def parse_spring_http_controller(captures: List[Tuple[Any, str]], source: str) -> List[Dict[str, Any]]:
    """从 spring_http_controller 查询结果中解析出 Spring Controller 类。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "controller_class":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue
        seen.add(node_id)

        # 检查该类是否带有 RestController / Controller 注解（在 modifiers 下的 annotation）
        has_controller_ann = False
        for child in node.children:
            if child.type != "modifiers":
                continue
            for mod in child.children:
                ann_name_node = None
                if mod.type == "marker_annotation":
                    ann_name_node = mod.child_by_field_name("name")
                elif mod.type == "annotation":
                    ann_name_node = mod.child_by_field_name("name")
                if ann_name_node:
                    name = _annotation_simple_name(_get_node_text(ann_name_node))
                    if name in SPRING_CONTROLLER_ANNOTATIONS:
                        has_controller_ann = True
                        break
            if has_controller_ann:
                break

        if not has_controller_ann:
            continue

        class_level_path = _get_class_level_request_mapping_path(node, _get_node_text)
        name_node = node.child_by_field_name("name")
        class_name = _get_node_text(name_node) if name_node else ""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        result.append({
            "kind": "spring_http_controller",
            "name": class_name,
            "line_number": start_line,
            "end_line": end_line,
            "path": "",
            "lang": "java",
            "class_level_path": class_level_path,
        })
    return result


def parse_spring_http_endpoint_method(
    captures: List[Tuple[Any, str]], source: str
) -> List[Dict[str, Any]]:
    """从 spring_http_endpoint_method 查询结果中解析出 Spring 端点方法。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "endpoint_method":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        mapping_ann_node: Optional[Any] = None
        mapping_ann_name = ""
        for child in node.children:
            if child.type != "modifiers":
                continue
            for mod in child.children:
                ann_name_node = None
                if mod.type == "marker_annotation":
                    ann_name_node = mod.child_by_field_name("name")
                elif mod.type == "annotation":
                    ann_name_node = mod.child_by_field_name("name")
                if ann_name_node:
                    name = _annotation_simple_name(_get_node_text(ann_name_node))
                    if name in SPRING_MAPPING_ANNOTATIONS:
                        mapping_ann_node = mod
                        mapping_ann_name = name
                        break
            if mapping_ann_node is not None:
                break

        if mapping_ann_node is None:
            continue
        seen.add(node_id)

        method_path, http_method = "", "GET"
        if mapping_ann_node:
            method_path, http_method = _extract_annotation_path_and_method(
                mapping_ann_node, mapping_ann_name, _get_node_text
            )

        class_level_path = ""
        parent_class = _find_ancestor(node, "class_declaration")
        if parent_class:
            class_level_path = _get_class_level_request_mapping_path(parent_class, _get_node_text)
        full_path = f"{http_method} {_join_path(class_level_path, method_path)}".strip()

        name_node = node.child_by_field_name("name")
        method_name = _get_node_text(name_node) if name_node else ""
        params_node = node.child_by_field_name("parameters")
        params_text = _get_node_text(params_node) if params_node else "()"
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        # 可选：解析参数名列表（与 java.py 中逻辑一致可复用）
        parameters: List[str] = []
        if params_text and params_text.strip() != "()":
            inner = params_text.strip("()").strip()
            if inner:
                for part in inner.split(","):
                    part = part.strip()
                    if part:
                        tokens = part.split()
                        if len(tokens) >= 2:
                            parameters.append(tokens[-1])

        controller_name: Optional[str] = None
        if parent_class:
            pn = parent_class.child_by_field_name("name")
            if pn:
                controller_name = _get_node_text(pn)

        result.append({
            "kind": "spring_http_endpoint",
            "name": method_name,
            "parameters": parameters,
            "line_number": start_line,
            "end_line": end_line,
            "path": "",
            "lang": "java",
            "controller_context": controller_name,
            "http_method": http_method,
            "endpoint_path": method_path,
            "full_path": full_path,
        })
    return result


def parse_feign_client_interface(captures: List[Tuple[Any, str]], source: str) -> List[Dict[str, Any]]:
    """从 feign_client_interface 查询结果中解析出 @FeignClient 接口。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "feign_interface":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        has_feign = False
        for child in node.children:
            if child.type != "modifiers":
                continue
            for mod in child.children:
                ann_name_node = None
                if mod.type == "marker_annotation":
                    ann_name_node = mod.child_by_field_name("name")
                elif mod.type == "annotation":
                    ann_name_node = mod.child_by_field_name("name")
                if ann_name_node:
                    name = _annotation_simple_name(_get_node_text(ann_name_node))
                    if name == FEIGN_CLIENT_ANNOTATION:
                        has_feign = True
                        break
            if has_feign:
                break
        if not has_feign:
            continue
        seen.add(node_id)

        base_url, service_name = _extract_feign_client_base(node, _get_node_text)
        name_node = node.child_by_field_name("name")
        interface_name = _get_node_text(name_node) if name_node else ""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        result.append({
            "kind": "feign_client_interface",
            "name": interface_name,
            "line_number": start_line,
            "end_line": end_line,
            "path": "",
            "lang": "java",
            "base_url": base_url,
            "service_name": service_name,
        })
    return result


def parse_feign_client_method(captures: List[Tuple[Any, str]], source: str) -> List[Dict[str, Any]]:
    """从 feign_client_method 查询结果中解析出 Feign 接口中的方法。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "feign_method":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        # 确认所在 interface 带 @FeignClient
        interface_node = _find_ancestor(node, "interface_declaration")
        if not interface_node:
            continue
        has_feign = False
        for child in interface_node.children:
            if child.type != "modifiers":
                continue
            for mod in child.children:
                ann_name_node = None
                if mod.type == "marker_annotation":
                    ann_name_node = mod.child_by_field_name("name")
                elif mod.type == "annotation":
                    ann_name_node = mod.child_by_field_name("name")
                if ann_name_node:
                    name = _annotation_simple_name(_get_node_text(ann_name_node))
                    if name == FEIGN_CLIENT_ANNOTATION:
                        has_feign = True
                        break
            if has_feign:
                break
        if not has_feign:
            continue
        seen.add(node_id)

        # 方法上的映射注解（GetMapping / PostMapping / RequestMapping 等）
        mapping_ann_node = None
        mapping_ann_name = ""
        for child in node.children:
            if child.type != "modifiers":
                continue
            for mod in child.children:
                ann_name_node = None
                if mod.type == "marker_annotation":
                    ann_name_node = mod.child_by_field_name("name")
                elif mod.type == "annotation":
                    ann_name_node = mod.child_by_field_name("name")
                if ann_name_node:
                    name = _annotation_simple_name(_get_node_text(ann_name_node))
                    if name in SPRING_MAPPING_ANNOTATIONS:
                        mapping_ann_node = mod
                        mapping_ann_name = name
                        break
            if mapping_ann_node is not None:
                break

        method_path, http_method = "", "GET"
        if mapping_ann_node:
            method_path, http_method = _extract_annotation_path_and_method(
                mapping_ann_node, mapping_ann_name, _get_node_text
            )

        if_node = _find_ancestor(node, "interface_declaration")
        interface_name = None
        class_level_path = ""
        base_url = ""
        service_name = ""
        if if_node:
            nn = if_node.child_by_field_name("name")
            if nn:
                interface_name = _get_node_text(nn)
            class_level_path = _get_class_level_request_mapping_path(if_node, _get_node_text)
            base_url, service_name = _extract_feign_client_base(if_node, _get_node_text)

        full_path = f"{http_method} {_join_path(class_level_path, method_path)}".strip()

        name_node = node.child_by_field_name("name")
        method_name = _get_node_text(name_node) if name_node else ""
        params_node = node.child_by_field_name("parameters")
        params_text = _get_node_text(params_node) if params_node else "()"
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        parameters = []
        if params_text and params_text.strip() != "()":
            inner = params_text.strip("()").strip()
            if inner:
                for part in inner.split(","):
                    part = part.strip()
                    if part:
                        tokens = part.split()
                        if len(tokens) >= 2:
                            parameters.append(tokens[-1])

        result.append({
            "kind": "feign_client_method",
            "name": method_name,
            "parameters": parameters,
            "line_number": start_line,
            "end_line": end_line,
            "path": "",
            "lang": "java",
            "interface_context": interface_name,
            "http_method": http_method,
            "endpoint_path": method_path,
            "full_path": full_path,
            "base_url": base_url,
            "service_name": service_name,
        })
    return result


def parse_grpc_service_class(captures: List[Tuple[Any, str]], source: str) -> List[Dict[str, Any]]:
    """从 grpc_service_class 查询结果中解析出 gRPC 服务实现类（继承 *Grpc.ServiceImplBase 等）。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "grpc_service_class":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        superclass_node = node.child_by_field_name("superclass")
        if not superclass_node:
            continue
        superclass_text = _get_node_text(superclass_node)
        if not any(hint in superclass_text for hint in GRPC_SERVICE_SUPERCLASS_HINTS):
            continue
        seen.add(node_id)

        name_node = node.child_by_field_name("name")
        class_name = _get_node_text(name_node) if name_node else ""
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        result.append({
            "kind": "grpc_service_class",
            "name": class_name,
            "line_number": start_line,
            "end_line": end_line,
            "path": "",
            "lang": "java",
            "superclass": superclass_text,
        })
    return result


def parse_grpc_service_endpoint_method(
    captures: List[Tuple[Any, str]], source: str
) -> List[Dict[str, Any]]:
    """从 grpc_service_endpoint_method 查询结果中解析出 gRPC 服务端点方法（服务实现中的 rpc 方法）。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "grpc_service_endpoint":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        service_class = _find_ancestor(node, "class_declaration")
        if not service_class:
            continue
        superclass_node = service_class.child_by_field_name("superclass")
        if not superclass_node:
            continue
        superclass_text = _get_node_text(superclass_node)
        if not any(hint in superclass_text for hint in GRPC_SERVICE_SUPERCLASS_HINTS):
            continue

        # 分层判定服务端点（见 _is_grpc_endpoint_method）
        params_node = node.child_by_field_name("parameters")
        params_text = _get_node_text(params_node) if params_node else "()"
        method_name_node = node.child_by_field_name("name")
        method_name = _get_node_text(method_name_node) if method_name_node else ""
        is_endpoint, endpoint_reason = _is_grpc_endpoint_method(
            node, method_name, params_text, _get_node_text
        )
        if not is_endpoint:
            continue

        seen.add(node_id)

        service_name_node = service_class.child_by_field_name("name")
        service_name = _get_node_text(service_name_node) if service_name_node else ""

        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1

        parameters: List[str] = []
        if params_text and params_text.strip() != "()":
            inner = params_text.strip("()").strip()
            if inner:
                for part in inner.split(","):
                    part = part.strip()
                    if part:
                        tokens = part.split()
                        if len(tokens) >= 2:
                            parameters.append(tokens[-1])

        result.append({
            "kind": "grpc_service_endpoint",
            "name": method_name,
            "parameters": parameters,
            "line_number": start_line,
            "end_line": end_line,
            "path": "",
            "lang": "java",
            "service_context": service_name,
            "superclass": superclass_text,
            "endpoint_reason": endpoint_reason,
        })
    return result


def parse_grpc_client_call(captures: list[tuple[Node, str]], source: str) -> List[Dict[str, Any]]:
    """从 grpc_client_call 查询结果中解析出 gRPC 客户端调用（*Stub.xxx()）。"""
    result: list[dict[str, Any]] = []
    seen: set[tuple[int,int]] = set()

    for node, capture_name in captures:
        if capture_name != "grpc_call":
            continue
        # node 即为 method_invocation（查询中已 @grpc_call）
        inv_node = node
        if not inv_node:
            continue
        node_id = (inv_node.start_byte, inv_node.end_byte)
        if node_id in seen:
            continue

        object_node = inv_node.child_by_field_name("object")
        if not object_node:
            continue
        receiver_text = _get_node_text(object_node)
        # 支持 "blockingStub" 或 "this.blockingStub" -> 取最后一段
        if "." in receiver_text:
            receiver_text = receiver_text.split(".")[-1]
        if not receiver_text.endswith(GRPC_CLIENT_STUB_SUFFIX):
            continue
        seen.add(node_id)

        name_node = inv_node.child_by_field_name("name")
        method_name = _get_node_text(name_node) if name_node else ""
        line = inv_node.start_point[0] + 1
        result.append({
            "kind": "grpc_client_call",
            "name": method_name,
            "receiver": receiver_text,
            "line_number": line,
            "path": "",
            "lang": "java",
        })
    return result


def parse_grpc_stub_class(captures: List[Tuple[Any, str]], source: str) -> List[Dict[str, Any]]:
    """从 grpc_stub_class 查询结果中解析 gRPC Stub 类（生成代码常见）。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "grpc_stub_class":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        is_stub, reason = _is_grpc_stub_class(node, _get_node_text)
        if not is_stub:
            continue
        seen.add(node_id)

        name_node = node.child_by_field_name("name")
        class_name = _get_node_text(name_node) if name_node else ""
        superclass_node = node.child_by_field_name("superclass")
        superclass_text = _get_node_text(superclass_node) if superclass_node else ""
        outer_class = _find_ancestor(node, "class_declaration")
        outer_name = ""
        if outer_class:
            outer_name_node = outer_class.child_by_field_name("name")
            outer_name = _get_node_text(outer_name_node) if outer_name_node else ""

        result.append({
            "kind": "grpc_stub_class",
            "name": class_name,
            "line_number": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "path": "",
            "lang": "java",
            "superclass": superclass_text,
            "match_reason": reason,
            "enclosing_grpc_context": outer_name,
        })
    return result


def parse_grpc_stub_method(captures: List[Tuple[Any, str]], source: str) -> List[Dict[str, Any]]:
    """从 grpc_stub_method 查询结果中解析位于 gRPC Stub 类中的方法。"""
    result: List[Dict[str, Any]] = []
    seen: set = set()

    for node, capture_name in captures:
        if capture_name != "grpc_stub_method":
            continue
        node_id = (node.start_byte, node.end_byte)
        if node_id in seen:
            continue

        stub_class = _find_ancestor(node, "class_declaration")
        if not stub_class:
            continue
        is_stub, reason = _is_grpc_stub_class(stub_class, _get_node_text)
        if not is_stub:
            continue

        name_node = node.child_by_field_name("name")
        method_name = _get_node_text(name_node) if name_node else ""
        if not method_name or method_name in GRPC_STUB_NON_RPC_METHOD_NAMES:
            continue
        seen.add(node_id)

        params_node = node.child_by_field_name("parameters")
        params_text = _get_node_text(params_node) if params_node else "()"
        parameters: List[str] = []
        if params_text and params_text.strip() != "()":
            inner = params_text.strip("()").strip()
            if inner:
                for part in inner.split(","):
                    part = part.strip()
                    if part:
                        tokens = part.split()
                        if len(tokens) >= 2:
                            parameters.append(tokens[-1])

        stub_name_node = stub_class.child_by_field_name("name")
        stub_name = _get_node_text(stub_name_node) if stub_name_node else ""

        result.append({
            "kind": "grpc_stub_method",
            "name": method_name,
            "parameters": parameters,
            "line_number": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
            "path": "",
            "lang": "java",
            "stub_context": stub_name,
            "stub_reason": reason,
        })
    return result


def run_java_rpc_queries(language: Any, tree: Any, source: str, path: Optional[Path] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    对已解析的 Java tree 执行 JAVA_RPC_QUERIES，并返回按类型聚合的 RPC 结果。

    Args:
        language: tree-sitter Language (java)
        tree: 已解析的 tree (tree.root_node 的父节点可为 tree)
        source: 源码字符串（当前解析函数未用，保留供后续扩展）
        path: 文件路径，用于回填 result 中的 "path"

    Returns:
        {
            "spring_http_controllers": [...],
            "spring_http_endpoints": [...],
            "feign_interfaces": [...],
            "feign_methods": [...],
            "grpc_service_classes": [...],
            "grpc_service_endpoints": [...],
            "grpc_client_calls": [...],
            "grpc_stub_classes": [...],
            "grpc_stub_methods": [...],
        }
    """
    root = tree.root_node if hasattr(tree, "root_node") else tree
    path_str = str(path) if path else ""

    parsers = [
        ("spring_http_controller", parse_spring_http_controller),
        ("spring_http_endpoint_method", parse_spring_http_endpoint_method),
        ("feign_client_interface", parse_feign_client_interface),
        ("feign_client_method", parse_feign_client_method),
        ("grpc_service_class", parse_grpc_service_class),
        ("grpc_service_endpoint_method", parse_grpc_service_endpoint_method),
        ("grpc_client_call", parse_grpc_client_call),
        ("grpc_stub_class", parse_grpc_stub_class),
        ("grpc_stub_method", parse_grpc_stub_method),
    ]

    out: Dict[str, List[Dict[str, Any]]] = {
        "spring_http_controllers": [],
        "spring_http_endpoints": [],
        "feign_interfaces": [],
        "feign_methods": [],
        "grpc_service_classes": [],
        "grpc_service_endpoints": [],
        "grpc_client_calls": [],
        "grpc_stub_classes": [],
        "grpc_stub_methods": [],
    }

    key_map = {
        "spring_http_controller": "spring_http_controllers",
        "spring_http_endpoint_method": "spring_http_endpoints",
        "feign_client_interface": "feign_interfaces",
        "feign_client_method": "feign_methods",
        "grpc_service_class": "grpc_service_classes",
        "grpc_service_endpoint_method": "grpc_service_endpoints",
        "grpc_client_call": "grpc_client_calls",
        "grpc_stub_class": "grpc_stub_classes",
        "grpc_stub_method": "grpc_stub_methods",
    }

    for query_key, parse_fn in parsers:
        query_str = JAVA_RPC_QUERIES.get(query_key)
        if not query_str:
            continue
        try:
            captures = execute_query(language, query_str, root)
            items = parse_fn(captures, source)
            for item in items:
                item["path"] = path_str
            out[key_map[query_key]] = items
        except Exception:
            # 单条查询失败不影响其余
            continue

    return out
