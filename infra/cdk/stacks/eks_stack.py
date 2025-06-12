from aws_cdk import (
    Stack,
    aws_eks as eks,
    aws_ec2 as ec2,
    aws_iam as iam,
    CfnOutput,
)
from constructs import Construct
import os
import yaml

class EksStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Create a VPC for the EKS cluster
        vpc = ec2.Vpc(self, "McpVpc",
            max_azs=3,
            nat_gateways=1
        )

        # Create an IAM role for the EKS cluster
        cluster_role = iam.Role(self, "ClusterRole",
            assumed_by=iam.ServicePrincipal("eks.amazonaws.com")
        )

        # Add required policies to the cluster role
        cluster_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSClusterPolicy")
        )

        # Create the EKS cluster
        cluster = eks.Cluster(self, "McpEksCluster",
            version=eks.KubernetesVersion.V1_32,
            vpc=vpc,
            default_capacity=0,  # We'll use managed node groups instead
            cluster_name="mcp-eks",
            role=cluster_role,
        )

        # Create an IAM role for the node group
        node_role = iam.Role(self, "NodeGroupRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com")
        )

        # Add required policies to the node role
        node_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKSWorkerNodePolicy")
        )
        node_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEKS_CNI_Policy")
        )
        node_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonEC2ContainerRegistryReadOnly")
        )

        # Create a managed node group with auto-scaling
        node_group = cluster.add_nodegroup_capacity("ManagedNodeGroup",
            instance_types=[ec2.InstanceType("m5.large")],
            min_size=2,
            max_size=10,
            desired_size=3,
            node_role=node_role,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            capacity_type=eks.CapacityType.ON_DEMAND,
        )

        # Create namespaces
        cluster.add_manifest("McpApplicationNamespace", {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": "mcp-application"}
        })

        cluster.add_manifest("McpServerNamespace", {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": "mcp-server"}
        })

        # Apply all YAML files from the eks directory
        self._apply_yaml_files(cluster, "../../../eks")

        # Export the cluster name and other outputs
        self.cluster = cluster

        CfnOutput(self, "ClusterName",
            value=cluster.cluster_name,
            description="Name of the EKS cluster"
        )

        CfnOutput(self, "ClusterEndpoint",
            value=cluster.cluster_endpoint,
            description="Endpoint for the EKS cluster"
        )

    def _apply_yaml_files(self, cluster, directory):
        """Apply all YAML files in the directory and its subdirectories to the cluster."""
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(".yaml"):
                    file_path = os.path.join(root, file)
                    self._apply_yaml_file(cluster, file_path)

    def _apply_yaml_file(self, cluster, file_path):
        """Apply a single YAML file to the cluster."""
        with open(file_path, 'r') as f:
            yaml_content = f.read()

        # Split the YAML file into individual documents (separated by ---)
        documents = yaml_content.split('---')

        for i, doc in enumerate(documents):
            if not doc.strip():
                continue  # Skip empty documents

            manifest = yaml.safe_load(doc)
            if manifest:
                # Create a unique name for each manifest
                base_name = os.path.basename(file_path).replace('.yaml', '')
                manifest_name = f"{base_name}-{i}"

                # Apply the manifest to the cluster
                cluster.add_manifest(manifest_name, manifest)