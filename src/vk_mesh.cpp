
#include "vk_mesh.h"
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include "iostream"

VertexInputDescription Vertex::getVertexDescription()
{
	VertexInputDescription description;

	VkVertexInputBindingDescription mainBinding = {};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

	description.bindings.push_back(mainBinding);

	//position at location 0
	VkVertexInputAttributeDescription positionAttr = {};
	positionAttr.binding = 0;
	positionAttr.location = 0;
	positionAttr.offset = offsetof(Vertex, position);
	positionAttr.format = VK_FORMAT_R32G32B32_SFLOAT;

	//normal at location 2
	VkVertexInputAttributeDescription normalAttr = {};
	normalAttr.binding = 0;
	normalAttr.location = 1;
	normalAttr.offset = offsetof(Vertex, normal);
	normalAttr.format = VK_FORMAT_R32G32B32_SFLOAT;

	//color at location 3
	VkVertexInputAttributeDescription colorAttr = {};
	colorAttr.binding = 0;
	colorAttr.location = 2;
	colorAttr.offset = offsetof(Vertex, color);
	colorAttr.format = VK_FORMAT_R32G32B32_SFLOAT;

	description.attributes.push_back(positionAttr);
	description.attributes.push_back(normalAttr);
	description.attributes.push_back(colorAttr);
	
	return description;
}

bool Mesh::loadFromObj(const char* filename)
{
	tinyobj::attrib_t attrib; //vtx array
	std::vector<tinyobj::shape_t> shapes; // separate objects
	std::vector<tinyobj::material_t> materials; //separate materials
	std::string warn;
	std::string err;

	tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename, nullptr);
	if (!warn.empty())
	{
		std::cout << "WARN: " << warn << std::endl;
	}
	if (!err.empty())
	{
		std::cerr << err << std::endl;
		return false;

	}

	//loop over all shapes
	for (size_t s = 0; s < shapes.size(); s++)
	{
		size_t index_offset = 0;

		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++)
		{
			int fv = 3;

			for (size_t v = 0; v < fv; v++)
			{
				//vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

				//vertex pos
				tinyobj::real_t vx = attrib.vertices[3 * idx.vertex_index + 0];
				tinyobj::real_t vy = attrib.vertices[3 * idx.vertex_index + 1];
				tinyobj::real_t vz = attrib.vertices[3 * idx.vertex_index + 2];

				//Vertex normal
				tinyobj::real_t nx = attrib.normals[3 * idx.vertex_index + 0];
				tinyobj::real_t ny = attrib.normals[3 * idx.vertex_index + 1];
				tinyobj::real_t nz = attrib.normals[3 * idx.vertex_index + 2];

				//copy it into our vertex
				Vertex new_vert;
				new_vert.position.x = vx;
				new_vert.position.y = vy; 
				new_vert.position.z = vz;

				new_vert.normal.x = nx;
				new_vert.normal.y = ny;
				new_vert.normal.z = nz;

				new_vert.color = new_vert.normal;

				_vertices.push_back(new_vert);
			}

			index_offset += fv;
		}
	}

	return true;
}
