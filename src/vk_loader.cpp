
#include <vk_loader.h>

#include "stb_image.h"
#include <iostream>
#include "vk_engine.h"
#include "glm/gtx/quaternion.hpp"

#include <fastgltf/glm_element_traits.hpp>
#include <fastgltf/parser.hpp>
#include <fastgltf/tools.hpp>

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath)
{
    std::cout << "Loading GLTF" << filePath << std::endl;

    fastgltf::GltfDataBuffer data;
    data.loadFromFile(filePath);

    constexpr auto gltfOptions = fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers;

    fastgltf::Asset gltf;
    fastgltf::Parser parser {};

    auto load = parser.loadBinaryGLTF(&data, filePath.parent_path(), gltfOptions);
    if (load)
    {
        gltf = std::move(load.get());
    }
    else
    {
        fmt::print("failed to load gltf {} \n", fastgltf::to_underlying(load.error()));
        return {};
    }

    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh& mesh : gltf.meshes)
    {
        MeshAsset newMesh;
        newMesh.name = mesh.name;

        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives)
        {

            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initialVtx = vertices.size();

            //load indexes
            {
                fastgltf::Accessor& indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexAccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + initialVtx);
                    });
            }

            //load vertices
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->second];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = { 1, 0, 0 };
                        newvtx.color = glm::vec4{ 1.f };
                        newvtx.UV = { 0,0 };
                        vertices[initialVtx + index] = newvtx;
                    });
            }

            //Load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).second],
                    [&](glm::vec3 v, size_t index) {
                        vertices[initialVtx + index].normal = v;
                    });
            }

            //load uvs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).second],
                    [&](glm::vec2 v, size_t index) {
                        vertices[initialVtx + index].UV = { v.x, v.y };;
                    });
            }

            //load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).second],
                    [&](glm::vec4 v, size_t index) {
                        vertices[initialVtx + index].color = v;
                    });
            }
            newMesh.surfaces.push_back(newSurface);
        }

        // display the vertex normals
        constexpr bool OverrideColors = true;
        if (OverrideColors) {
            for (Vertex& vtx : vertices) {
                vtx.color = glm::vec4(vtx.normal, 1.f);
            }
        }
        newMesh.meshBuffers = engine->uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
    }

    return meshes;
}
